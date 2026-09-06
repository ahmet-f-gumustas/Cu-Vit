#!/usr/bin/env python3
"""Export a timm Vision Transformer checkpoint to the Cu-Vit weight format.

The format is the one src/runtime/weights.cpp reads:

    magic u32, version u32, entry_count u32, reserved u32
    entry_count x {
        name_length u32, name bytes, type u32, rank u32, extents i64[rank],
        offset u64, bytes u64
    }
    payloads, each aligned to 64 bytes

Every tensor is stored as contiguous little-endian float32 in the layout timm
already uses, so the engine reads it without transposing anything at load time.

Usage:
    python3 tools/export_vit_weights.py --model vit_tiny_patch16_224 --output vit_tiny.cvw
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

MAGIC = 0x54495643  # "CVIT"
VERSION = 1
DTYPE_FLOAT32 = 1
PAYLOAD_ALIGNMENT = 64
MAX_RANK = 4


def align_up(value: int) -> int:
    return (value + PAYLOAD_ALIGNMENT - 1) // PAYLOAD_ALIGNMENT * PAYLOAD_ALIGNMENT


def collect_tensors(model):
    """Return [(name, shape, float32 bytes)] in a stable order.

    Sorting by name keeps two exports of the same checkpoint byte-identical,
    which makes the output diffable and cacheable.
    """
    import torch

    tensors = []
    for name, parameter in sorted(model.state_dict().items()):
        if not isinstance(parameter, torch.Tensor):
            continue
        if parameter.ndim == 0 or parameter.ndim > MAX_RANK:
            raise SystemExit(
                f"tensor '{name}' has rank {parameter.ndim}, which the format does not carry"
            )
        values = parameter.detach().to(torch.float32).contiguous().cpu()
        tensors.append((name, tuple(values.shape), values.numpy().tobytes()))
    return tensors


def write_weight_file(path: Path, tensors) -> None:
    table_bytes = 4 * 4
    for name, shape, _ in tensors:
        table_bytes += 4 + len(name.encode("utf-8"))
        table_bytes += 4 + 4
        table_bytes += 8 * len(shape)
        table_bytes += 8 + 8

    offsets = []
    cursor = align_up(table_bytes)
    for _, _, payload in tensors:
        offsets.append(cursor)
        cursor = align_up(cursor + len(payload))

    with path.open("wb") as handle:
        handle.write(struct.pack("<IIII", MAGIC, VERSION, len(tensors), 0))
        for (name, shape, payload), offset in zip(tensors, offsets):
            encoded = name.encode("utf-8")
            handle.write(struct.pack("<I", len(encoded)))
            handle.write(encoded)
            handle.write(struct.pack("<II", DTYPE_FLOAT32, len(shape)))
            for extent in shape:
                handle.write(struct.pack("<q", extent))
            handle.write(struct.pack("<QQ", offset, len(payload)))

        for (_, _, payload), offset in zip(tensors, offsets):
            padding = offset - handle.tell()
            if padding < 0:
                raise SystemExit("internal error: payload offsets overlap the table")
            handle.write(b"\0" * padding)
            handle.write(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="vit_tiny_patch16_224", help="timm model name")
    parser.add_argument("--output", type=Path, required=True, help="destination .cvw file")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="export randomly initialized weights instead of downloading a checkpoint",
    )
    arguments = parser.parse_args()

    try:
        import timm
    except ImportError:
        raise SystemExit("timm is required: pip install timm")

    model = timm.create_model(arguments.model, pretrained=not arguments.no_pretrained)
    model.eval()

    tensors = collect_tensors(model)
    write_weight_file(arguments.output, tensors)

    total = sum(len(payload) for _, _, payload in tensors)
    print(f"model:   {arguments.model}")
    print(f"tensors: {len(tensors)}")
    print(f"params:  {total // 4:,}")
    print(f"wrote:   {arguments.output} ({arguments.output.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
