#!/usr/bin/env python3
"""Normalize the Unsloth Qwen3.6-35B-A3B UD-Q4_K_S GGUF into the DS4 artifact.

Reproduces the qualified-artifact provenance documented in tests/qwen/README.md:
  - converts the Q6_K `ffn_down_exps` banks to the uniform Q4_K cache layout
  - converts `output.weight` from Q6_K to Q8_0
  - restores the official padding token ID (248044)
  - restores the canonical chat template (qwen36_chat_template.jinja)
Every other tensor payload is copied byte-identical (the container is
rewritten, so absolute offsets may differ; DS4 validates inventory and
payload sizes, not file offsets).

Requantization uses ggml's reference quantizers (ggml_quantize_chunk) via
ctypes — the same code path llama.cpp uses — so the produced Q4_K/Q8_0
blocks follow the reference layout DS4 validates.

Usage:
  normalize_qwen36_gguf.py SRC.gguf DST.gguf \
      --ggml-lib llama.cpp/build-cpu/bin/libggml-base.so \
      --gguf-py llama.cpp/gguf-py \
      --chat-template tests/qwen/qwen36_chat_template.jinja
"""

import argparse
import ctypes
import sys
from pathlib import Path

EXPECTED_PAD_ID = 248044
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q4_K = 12


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--ggml-lib", required=True)
    ap.add_argument("--gguf-py", required=True)
    ap.add_argument("--chat-template", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.gguf_py)
    import gguf  # noqa: E402
    from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType  # noqa: E402
    from gguf.constants import GGUFValueType  # noqa: E402
    from gguf.quants import quant_shape_from_byte_shape  # noqa: E402
    import numpy as np  # noqa: E402

    ggml = ctypes.CDLL(args.ggml_lib)
    ggml.ggml_quantize_chunk.restype = ctypes.c_size_t
    ggml.ggml_quantize_chunk.argtypes = [
        ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float),
    ]

    template = Path(args.chat_template).read_text(encoding="utf-8")
    reader = GGUFReader(args.src)

    arch = str(reader.get_field("general.architecture").contents())
    if arch != "qwen35moe":
        print(f"unexpected architecture {arch!r}", file=sys.stderr)
        return 1

    writer = GGUFWriter(args.dst, arch, endianess=reader.endianess)

    # --- metadata: mirror gguf_new_metadata.py's copy loop ---
    pad_key = "tokenizer.ggml.padding_token_id"
    saw_pad = False
    for field in reader.fields.values():
        if (field.name == "general.architecture"
                or field.name.startswith("GGUF.")):
            continue
        if field.name.startswith("tokenizer.chat_template"):
            continue  # replaced below with the canonical template
        val_type = field.types[0]
        sub_type = (field.types[-1]
                    if val_type == GGUFValueType.ARRAY else None)
        value = field.contents()
        if field.name == pad_key:
            print(f"padding_token_id: {value} -> {EXPECTED_PAD_ID}")
            value = EXPECTED_PAD_ID
            saw_pad = True
        writer.add_key_value(field.name, value, val_type, sub_type=sub_type)
    if not saw_pad:
        print(f"padding_token_id: absent -> {EXPECTED_PAD_ID}")
        writer.add_key_value(pad_key, EXPECTED_PAD_ID, GGUFValueType.UINT32)
    writer.add_chat_template(template)

    # --- identify the Q6_K set to convert (fail closed) ---
    q6_names = sorted(t.name for t in reader.tensors
                      if t.tensor_type == GGMLQuantizationType.Q6_K)
    bad = [n for n in q6_names
           if not (n.endswith("ffn_down_exps.weight") or n == "output.weight")]
    if bad or "output.weight" not in q6_names:
        print(f"unexpected Q6_K tensor set: {q6_names}", file=sys.stderr)
        return 1
    print(f"converting {len(q6_names)} Q6_K tensors: {q6_names}")

    def requantize(tensor, target: int):
        ttype = GGMLQuantizationType(target)
        block, tsize = gguf.constants.GGML_QUANT_SIZES[ttype]
        logical = quant_shape_from_byte_shape(tensor.data.shape,
                                              tensor.tensor_type)
        n_per_row = int(logical[-1])
        nrows = int(np.prod(logical[:-1], dtype=np.int64)) if len(logical) > 1 else 1
        f32 = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
        f32 = np.ascontiguousarray(f32, dtype=np.float32).reshape(nrows,
                                                                  n_per_row)
        row_bytes = (n_per_row // block) * tsize
        out = np.empty((*logical[:-1], row_bytes), dtype=np.uint8)
        written = ggml.ggml_quantize_chunk(
            target,
            f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.c_void_p),
            0, nrows, n_per_row, None)
        assert written == out.nbytes, (tensor.name, written, out.nbytes)
        return out, ttype

    # --- tensor info pass, then data pass (same order) ---
    converted = {}
    total = 0
    for t in reader.tensors:
        if t.tensor_type == GGMLQuantizationType.Q6_K:
            target = (GGML_TYPE_Q8_0 if t.name == "output.weight"
                      else GGML_TYPE_Q4_K)
            data, dtype = requantize(t, target)
            converted[t.name] = data
            print(f"  {t.name}: Q6_K -> {dtype.name} "
                  f"({t.n_bytes} -> {data.nbytes} bytes)")
            writer.add_tensor_info(t.name, data.shape, data.dtype,
                                   data.nbytes, dtype)
            total += data.nbytes
        else:
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype,
                                   t.data.nbytes, t.tensor_type)
            total += t.data.nbytes

    print(f"tensor payload total: {total} bytes over {len(reader.tensors)} tensors")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in reader.tensors:
        data = converted.get(t.name)
        writer.write_tensor_data(data if data is not None else t.data,
                                 tensor_endianess=reader.endianess)
    writer.close()
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
