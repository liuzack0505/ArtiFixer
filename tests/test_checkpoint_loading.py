# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import unittest
from unittest.mock import patch
from argparse import Namespace
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

from model_eval.checkpoint_loading import (
    add_checkpoint_args,
    checkpoint_output_name,
    load_transformer_checkpoint,
    validate_checkpoint_args,
)


class CheckpointLoadingTests(unittest.TestCase):
    def parse_checkpoint_args(self, argv):
        parser = argparse.ArgumentParser()
        add_checkpoint_args(parser)
        args = parser.parse_args(argv)
        validate_checkpoint_args(parser, args)
        return args

    def test_requires_exactly_one_checkpoint_source(self):
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                self.parse_checkpoint_args([])

            with self.assertRaises(SystemExit):
                self.parse_checkpoint_args(["--checkpoint_dir", "/tmp/dcp", "--checkpoint_pt", "/tmp/model.pt"])

    def test_preserves_dcp_output_name(self):
        args = self.parse_checkpoint_args(["--checkpoint_dir", "/runs/checkpoint_400/pytorch_model_fsdp_2"])

        self.assertEqual(checkpoint_output_name(args), "checkpoint_400_pytorch_model_fsdp_2")

    def test_checkpoint_pt_output_name_uses_file_stem(self):
        args = self.parse_checkpoint_args(
            ["--checkpoint_pt", "/exports/artifixer-s3-dmd-14b-sr2he0ue-ckpt400-fsdp2.pt"]
        )

        self.assertEqual(checkpoint_output_name(args), "artifixer-s3-dmd-14b-sr2he0ue-ckpt400-fsdp2")

    def test_load_transformer_checkpoint_loads_pt_tensors_to_target_device(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")

        model = torch.nn.Linear(2, 3, device="meta")
        state_dict = {
            "weight": torch.ones((3, 2), dtype=torch.float32),
            "bias": torch.arange(3, dtype=torch.float32),
        }

        with patch("model_eval.checkpoint_loading._torch_load_state_dict", return_value=state_dict):
            load_transformer_checkpoint(
                model,
                Namespace(checkpoint_dir=None, checkpoint_pt=Path("/exports/model.pt")),
                target_device=torch.device("cpu"),
                target_dtype=torch.float16,
            )

        self.assertFalse(model.weight.is_meta)
        self.assertFalse(model.bias.is_meta)
        self.assertEqual(model.weight.dtype, torch.float16)
        self.assertEqual(model.bias.dtype, torch.float16)
        torch.testing.assert_close(model.weight, torch.ones((3, 2), dtype=torch.float16))
        torch.testing.assert_close(model.bias, torch.arange(3, dtype=torch.float16))

    def test_load_transformer_checkpoint_dispatches_dcp_source(self):
        calls = []
        model = object()

        load_transformer_checkpoint(
            model,
            Namespace(checkpoint_dir=Path("/runs/checkpoint_400/pytorch_model_fsdp_2"), checkpoint_pt=None),
            dcp_loader=lambda loaded_model, path, **kwargs: calls.append(("dcp", loaded_model, path)),
        )

        self.assertEqual(
            calls,
            [("dcp", model, Path("/runs/checkpoint_400/pytorch_model_fsdp_2"))],
        )


if __name__ == "__main__":
    unittest.main()
