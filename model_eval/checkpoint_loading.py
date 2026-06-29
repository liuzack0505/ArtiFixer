# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from argparse import Namespace
from pathlib import Path


def add_checkpoint_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint_dir", type=Path, default=None)
    parser.add_argument(
        "--checkpoint_pt",
        type=Path,
        default=None,
        help="Single-file transformer state dict exported from a DCP/FSDP checkpoint.",
    )


def validate_checkpoint_args(parser: argparse.ArgumentParser, args: Namespace) -> None:
    has_checkpoint_dir = args.checkpoint_dir is not None
    has_checkpoint_pt = args.checkpoint_pt is not None
    if has_checkpoint_dir == has_checkpoint_pt:
        parser.error(
            "exactly one of --checkpoint_dir or --checkpoint_pt is required")


def checkpoint_output_name(args: Namespace) -> str:
    if args.checkpoint_pt is not None:
        return args.checkpoint_pt.stem

    ckpt_parent = args.checkpoint_dir.parent.name
    ckpt_name = args.checkpoint_dir.name
    return f"{ckpt_parent}_{ckpt_name}" if ckpt_parent.startswith("checkpoint") else ckpt_name


def _torch_load_state_dict(checkpoint_pt: Path, *, map_location):
    import torch

    try:
        return torch.load(checkpoint_pt, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_pt, map_location=map_location)


def load_model_weights_from_pt(model, checkpoint_pt: Path | str, *, target_device, target_dtype=None) -> None:
    import gc

    import torch

    if target_device is None:
        raise ValueError(
            "target_device is required when loading checkpoint weights from .pt")

    state_dict = _torch_load_state_dict(
        Path(checkpoint_pt), map_location=target_device)
    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Expected a state dict in {checkpoint_pt}, got {type(state_dict).__name__}")

    if target_dtype is not None:
        for name in list(state_dict):
            tensor = state_dict[name]
            if torch.is_tensor(tensor) and torch.is_floating_point(tensor) and tensor.dtype != target_dtype:
                state_dict[name] = tensor.to(dtype=target_dtype)
                del tensor

    try:
        model.load_state_dict(state_dict, strict=True, assign=True)
    except TypeError as e:
        raise RuntimeError(
            "Loading checkpoint tensors directly requires torch.nn.Module.load_state_dict(assign=True). "
            "Use a PyTorch version that supports assign=True."
        ) from e
    del state_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_transformer_checkpoint(
    model,
    args: Namespace,
    *,
    dcp_loader=None,
    target_device=None,
    target_dtype=None,
) -> None:
    if args.checkpoint_pt is not None:
        load_model_weights_from_pt(
            model,
            args.checkpoint_pt,
            target_device=target_device,
            target_dtype=target_dtype,
        )
        return

    if dcp_loader is None:
        from model_training.utils.train_utils import load_model_weights_from_dcp

        dcp_loader = load_model_weights_from_dcp
    dcp_loader(
        model,
        args.checkpoint_dir,
        target_device=target_device,
        target_dtype=target_dtype,
    )
