# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from model_eval.datasets.reconstructed_colmap_eval import ReconstructedColmapEvalDataset
from model_training.data.utils import (
    NeighborSelectionMode,
    gaussian_frustum_masks,
    select_neighbor_indices_gaussian_frustum,
)

# 64x64 pinhole with a 90 degree field of view.
INTRINSICS = (64.0, 64.0, 32.0, 32.0, 32.0, 32.0)


def camera_at(x: float, z: float = 10.0) -> list[list[float]]:
    """OpenGL camera-to-world at (x, 0, z) looking down -Z, i.e. toward the z=0 plane."""
    return [
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def unpack(masks: np.ndarray, count: int) -> np.ndarray:
    return np.unpackbits(masks, axis=1, count=count).astype(bool)


class GaussianFrustumMaskTests(unittest.TestCase):
    def test_mask_keeps_only_points_inside_the_frustum(self) -> None:
        positions = np.array(
            [
                [0.0, 0.0, 0.0],  # straight ahead
                [9.0, 0.0, 0.0],  # ahead, inside the 90 degree cone at depth 10
                [11.0, 0.0, 0.0],  # ahead, outside the cone
                [0.0, -9.0, 0.0],  # ahead, inside vertically
                [0.0, 0.0, 20.0],  # behind the camera
            ]
        )
        masks = gaussian_frustum_masks(positions, np.array([camera_at(0.0)]), np.array([INTRINSICS]))
        self.assertEqual(unpack(masks, len(positions))[0].tolist(), [True, True, False, True, False])

    def test_mask_clips_to_near_and_far_planes(self) -> None:
        # Camera at z=10 looking down -Z, so depth = 10 - z.
        depths = [0.005, 0.02, 999.0, 1001.0]
        positions = np.array([[0.0, 0.0, 10.0 - depth] for depth in depths])
        masks = gaussian_frustum_masks(positions, np.array([camera_at(0.0)]), np.array([INTRINSICS]))
        self.assertEqual(unpack(masks, len(positions))[0].tolist(), [False, True, True, False])

        custom = gaussian_frustum_masks(
            positions, np.array([camera_at(0.0)]), np.array([INTRINSICS]), near=0.001, far=500.0
        )
        self.assertEqual(unpack(custom, len(positions))[0].tolist(), [True, True, False, False])


class GaussianFrustumSelectionTests(unittest.TestCase):
    def test_ranks_train_views_by_shared_target_gaussians(self) -> None:
        # A row of Gaussians along x; each camera sees x within +-10 of its own x.
        positions = np.stack([np.arange(-40.0, 41.0), np.zeros(81), np.zeros(81)], axis=1)
        xs = [0.0, 2.0, 15.0, 35.0, -30.0]  # frame 0 is the target, frames 1-4 are train views
        masks = gaussian_frustum_masks(
            positions, np.array([camera_at(x) for x in xs]), np.array([INTRINSICS] * len(xs))
        )

        selected, _ = select_neighbor_indices_gaussian_frustum({1, 2, 3, 4}, [0], masks, num_train=2)

        # Frame 1 overlaps the target almost fully, frame 2 partially, frames 3 and 4 not at all.
        self.assertEqual(selected, [1, 2])

    def test_returns_empty_when_too_few_train_views(self) -> None:
        masks = np.zeros((3, 1), dtype=np.uint8)
        self.assertEqual(select_neighbor_indices_gaussian_frustum({1}, [0], masks, num_train=2), ([], False))


class GaussianFrustumDatasetTests(unittest.TestCase):
    def write_scene(self, root: Path, frame_xs: list[float], train_ids: list[int], target_ids: list[int]) -> Path:
        positions = torch.tensor([[x, 0.0, 0.0] for x in np.arange(-40.0, 41.0)], dtype=torch.float32)
        density = torch.full((len(positions), 1), 5.0)  # sigmoid(5) ~ 0.99, kept by the opacity filter
        density[positions[:, 0] > 20.0] = -5.0  # sigmoid(-5) ~ 0.01, ignored
        torch.save({"positions": positions, "density": density}, root / "ckpt.pt")

        w, h, fl_x, fl_y, cx, cy = INTRINSICS
        (root / "transforms.json").write_text(
            json.dumps(
                {
                    "w": int(w),
                    "h": int(h),
                    "fl_x": fl_x,
                    "fl_y": fl_y,
                    "cx": cx,
                    "cy": cy,
                    "frames": [
                        {"file_path": f"images/{index:05d}.png", "transform_matrix": camera_at(x)}
                        for index, x in enumerate(frame_xs)
                    ],
                }
            )
        )
        (root / "selected_indices.json").write_text(json.dumps(train_ids))
        (root / "target_indices.json").write_text(json.dumps(target_ids))
        for name in ("image_root", "renders", "opacity"):
            (root / name).mkdir()
        (root / "caption.h5").write_bytes(b"placeholder")
        split_path = root / "split.json"
        split_path.write_text(
            json.dumps(
                {
                    "test": {
                        "scene": {
                            "transforms_path": "transforms.json",
                            "image_root": "image_root",
                            "render_dir": "renders",
                            "opacity_dir": "opacity",
                            "selected_indices_path": "selected_indices.json",
                            "target_indices_path": "target_indices.json",
                            "prompt_path": "caption.h5",
                            "camera_scale": 1.0,
                            "reconstruction_checkpoint": "ckpt.pt",
                        }
                    }
                }
            )
        )
        return split_path

    def test_chunk_selection_uses_checkpoint_gaussians(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        # Targets 0-1 near x=0; train views 2-5. Frame 5 at x=30 only sees low-opacity Gaussians.
        split_path = self.write_scene(root, [0.0, 1.0, 3.0, -12.0, -35.0, 30.0], [2, 3, 4, 5], [0, 1])

        dataset = ReconstructedColmapEvalDataset(
            split="test",
            split_path=split_path,
            num_views=2,
            neighbor_selection_mode=NeighborSelectionMode.GAUSSIAN_FRUSTUM,
            use_target_indices=True,
        )

        self.assertEqual(len(dataset.inference_items), 1)
        self.assertEqual(dataset.inference_items[0][1].neighbor_indices, [2, 3])

    def test_latent_frame_selection_follows_each_temporal_group(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        # Target 0 looks at x=-30, targets 1-4 at x=+10; train view 5 matches the first, 6 the rest.
        split_path = self.write_scene(root, [-30.0, 10.0, 10.0, 10.0, 10.0, -30.0, 10.0], [5, 6], [0, 1, 2, 3, 4])

        dataset = ReconstructedColmapEvalDataset(
            split="test",
            split_path=split_path,
            num_views=1,
            neighbor_selection_mode=NeighborSelectionMode.GAUSSIAN_FRUSTUM,
            neighbor_selection_granularity="latent_frame",
            use_target_indices=True,
        )
        neighbor_indices, mask = dataset._select_latent_frame_neighbors("scene", [0, 1, 2, 3, 4], [True] * 5)

        selected = [[neighbor_indices[i] for i in row.nonzero().flatten().tolist()] for row in mask]
        self.assertEqual(selected, [[5], [6]])

    def test_requires_reconstruction_checkpoint(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        split_path = self.write_scene(root, [0.0, 3.0], [1], [0])
        (root / "ckpt.pt").unlink()

        with self.assertRaisesRegex(AssertionError, "reconstruction_checkpoint"):
            ReconstructedColmapEvalDataset(
                split="test",
                split_path=split_path,
                num_views=1,
                neighbor_selection_mode=NeighborSelectionMode.GAUSSIAN_FRUSTUM,
                use_target_indices=True,
            )


if __name__ == "__main__":
    unittest.main()
