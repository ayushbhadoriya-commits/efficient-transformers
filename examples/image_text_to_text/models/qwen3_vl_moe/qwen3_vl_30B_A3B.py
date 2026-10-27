# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Continuous-batching disaggregated prefill/decode for Qwen3-VL-MoE — DMA KV handoff."""

import argparse
from collections import deque
from time import perf_counter

import numpy as np
import requests
import torch
import transformers
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession

ENCODE_CONFIG = {
    "prefill_seq_len": 10240,
    "ctx_len": 10240,
    "batch_size": 1,
    "full_batch_size": 1,
    "num_devices": 1,
    "num_cores": 16,
    "mxfp6_matmul": False,
    "mxint8_kv_cache": False,
    "aic_enable_depth_first": False,
    "mos": 1,
    "prefill_only": None,
    "height": 512,
    "width": 910,
    "use_onnx_subfunctions": True,
    "split_model_io": True,
    "mm_processor_kwargs": {
        "max_pixels": 16777216,
        "min_pixels": 4096,
    },
    "skip_lang": True,
}

PREFILL_CONFIG = {
    "prefill_seq_len": 1024,
    "ctx_len": 10240,
    "batch_size": 1,
    "full_batch_size": 1,
    "num_devices": 7,
    "num_cores": 16,
    "mxfp6_matmul": True,
    "mxint8_kv_cache": True,
    "aic_enable_depth_first": False,
    "prefill_only": True,
    "vision_batch_size": 1,
    "height": 512,
    "width": 910,
    "split_model_io": True,
    "use_onnx_subfunctions": True,
    "enable_chunking": True,
    "retain_full_kv": True,
    "kv_cache_prefix": "vllmKvCache",
    "kv_cache_batch_size": 1,
    "layerwise": False,
    "user_tiled": True,
    "mdp_num_partitions": 7,
    "mdp_strategy": "intersection",
    "offload_pt_weights": False,
    "qaic_config": {
        "blocking_mode": "prefill_online",
        "num_kv_blocks": 32,
        "num_q_blocks": 1,
        # For the 30B model, NQH / num_cores is 2; n_rep_chunk=4 fails the
        # blocked-attention divisibility assertion.
        "n_rep_chunk": 2,
        "ctx_len": 10240,
        "skip_kv": True,
        "moe_config": {
            "flavour": "expert_parallel",
            "expert_parallel_chunk_size": 256,
            "cores_per_expert": 1,
            "tree_reduce": False,
        },
    },
    "allow_mxint8_mdp_io": True,
    "mm_processor_kwargs": {
        "max_pixels": 16777216,
        "min_pixels": 4096,
    },
    "skip_vision": True,
    "vision_size": 1024,
}

DECODE_CONFIG = {
    "prefill_seq_len": 1,
    "ctx_len": 10240,
    "batch_size": 256,
    "full_batch_size": 256,
    "num_devices": 16,
    "num_cores": 16,
    "mxfp6_matmul": True,
    "mxint8_kv_cache": True,
    "aic_enable_depth_first": False,
    "prefill_only": False,
    "vision_batch_size": 1,
    "height": 512,
    "width": 910,
    "retain_full_kv": True,
    "kv_cache_batch_size": 256,
    "split_model_io": True,
    "use_onnx_subfunctions": True,
    "kv_cache_prefix": "vllmKvCache",
    "layerwise": False,
    "user_tiled": True,
    "offload_pt_weights": True,
    "qaic_config": {
        "blocking_mode": "kv_batch_fold",
        "num_kv_blocks": 4,
        "ctx_len": 10240,
        "skip_kv": True,
        "moe_config": {
            "flavour": "expert_parallel",
            "expert_parallel_chunk_size": 256,
            "cores_per_expert": 2,
            "tree_reduce": False,
        },
    },
    "allow_mxint8_mdp_io": True,
    "mm_processor_kwargs": {
        "max_pixels": 16777216,
        "min_pixels": 4096,
    },
    "skip_vision": True,
    "vision_size": 1,
}

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"  # "Qwen/Qwen3-VL-235B-A22B-Instruct"
DEFAULT_PROMPTS = [
    "Tell me about yourself.",
    "What is the capital of France?",
    "Explain photosynthesis in one sentence.",
    "Name three primary colors.",
]
DEFAULT_IMAGE_PROMPTS = [
    # "Describe all the colors seen in the image",
    # "What are the objects in the image?",
    # "What is the main subject of the image?",
    "Describe the image in detail",
    "Describe the image in detail",
]
DEFAULT_IMAGE_URLS = [
    "https://picsum.photos/id/237/536/354",
    "https://picsum.photos/id/238/536/354",
    # "https://picsum.photos/id/234/536/354",
    # "https://picsum.photos/id/235/536/354",
]
DEFAULT_PREFILL_SEQ_LEN = PREFILL_CONFIG["prefill_seq_len"]
DEFAULT_CTX_LEN = PREFILL_CONFIG["ctx_len"]
DEFAULT_GENERATION_LEN = 100
DEFAULT_FULL_BATCH_SIZE = DECODE_CONFIG["full_batch_size"]

STAGES = PREFILL_CONFIG["mdp_num_partitions"]
PREFILL_NUM_DEVICES = PREFILL_CONFIG["num_devices"]
DECODE_NUM_DEVICES = DECODE_CONFIG["num_devices"]
VISION_INPUT_KEYS = {
    "pixel_values",
    "image_masks",
    "image_input_idx",
    "image_grid_thw",
    "valid_idx",
    "aspect_ratio_ids",
    "aspect_ratio_mask",
}
VISION_FP16_KEYS = {"pixel_values", "image_masks"}
VISION_OUTPUT_KEYS = ("vision_embeds", "deepstack_features")
# The encoder QPC is specialized for a 32 x 56 patch grid. Qwen3-VL uses a
# 16-pixel spatial patch, so runtime images must be 512 x 896 pixels.
VISION_IMAGE_HEIGHT = 512
VISION_IMAGE_WIDTH = 896


def run(
    model_id: str = DEFAULT_MODEL_ID,
    prompts=None,
    prefill_seq_len: int = DEFAULT_PREFILL_SEQ_LEN,
    ctx_len: int = DEFAULT_CTX_LEN,
    generation_len: int = DEFAULT_GENERATION_LEN,
    full_batch_size: int = DEFAULT_FULL_BATCH_SIZE,
    skip_vision: bool = False,
    image_urls=None,
    stages: int = STAGES,
    prefill_num_devices: int = PREFILL_NUM_DEVICES,
    decode_num_devices: int = DECODE_NUM_DEVICES,
    mdp_load_partition_config: str | None = None,
    num_hidden_layers: int | None = None,
    vision_depth: int | None = None,
    generate_mdp_only: bool = False,
    prefill_only: bool = False,
    compile_only: bool = False,
):
    """Run CB (chunked-prefill + batched decode) over ``prompts`` with the DMA KV handoff.

    When ``prefill_only`` is set, execute the initial prompt prefill and return
    before compiling or running the decode QPC.

    When ``compile_only`` is set, compile the prefill and decode QPCs and
    return before constructing runtime sessions or running inference.
    """
    prompts = list(prompts) if prompts else list(DEFAULT_PROMPTS)
    image_urls = list(image_urls) if image_urls else list(DEFAULT_IMAGE_URLS)
    if len(prompts) == 1:
        prompts = prompts * full_batch_size
    if len(image_urls) == 1:
        image_urls = image_urls * full_batch_size

    ########### CONFIG --------------------------
    config = AutoConfig.from_pretrained(model_id)
    config.dtype = "float16"
    config.torch_dtype = torch.float16

    # Test-only graph reduction.  The checkpoint architecture is preserved by
    # default; callers must explicitly opt into smaller layer counts.
    if num_hidden_layers is not None:
        config.text_config.num_hidden_layers = num_hidden_layers
    if vision_depth is not None:
        config.vision_config.depth = vision_depth
        config.vision_config.deepstack_visual_indexes = [vision_depth - 1]

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        model_id,
        attn_implementation="eager",
        kv_offload=True,
        continuous_batching=True,
        config=config,
        dtype=torch.float16,
        weight_free=True,
        layerwise=False,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)

    vision_session = None
    if not skip_vision:
        print(f">>>>>>>>>>>>>> ENCODE_CONFIG :{ENCODE_CONFIG}")
        vision_qpc_path = qeff_model.compile(**ENCODE_CONFIG)
        vision_session = QAICInferenceSession(
            vision_qpc_path.get("vision_qpc_path")
        )


    PREFILL_CONFIG["prefill_seq_len"] = prefill_seq_len
    PREFILL_CONFIG["ctx_len"] = ctx_len
    PREFILL_CONFIG["num_devices"] = prefill_num_devices
    PREFILL_CONFIG["mdp_num_partitions"] = stages
    PREFILL_CONFIG["qaic_config"]["ctx_len"] = ctx_len
    if mdp_load_partition_config:
        PREFILL_CONFIG["mdp_load_partition_config"] = mdp_load_partition_config
    DECODE_CONFIG["ctx_len"] = ctx_len
    DECODE_CONFIG["batch_size"] = full_batch_size
    DECODE_CONFIG["full_batch_size"] = full_batch_size
    DECODE_CONFIG["kv_cache_batch_size"] = full_batch_size
    DECODE_CONFIG["num_devices"] = decode_num_devices
    DECODE_CONFIG["qaic_config"]["ctx_len"] = ctx_len
    print(f">>>>>>>>>>>>>> PREFILL_CONFIG :{PREFILL_CONFIG}")
    prefill_qpc_path = qeff_model.compile(**PREFILL_CONFIG, artifacts=generate_mdp_only)

    if generate_mdp_only:
        print(f"MDP artifacts written to: {prefill_qpc_path}")
        print("Stopped after MDP generation (no final compile, decode compile, or inference).")
        return prefill_qpc_path

    if compile_only:
        decode_qpc_path = qeff_model.compile(**DECODE_CONFIG)
        print(f"Prefill QPC compiled at: {prefill_qpc_path}")
        print(f"Decode QPC compiled at: {decode_qpc_path}")
        print("Stopped after QPC compilation (no runtime sessions or inference).")
        return {"prefill": prefill_qpc_path, "decode": decode_qpc_path}


    prefill_session = QAICInferenceSession(
        prefill_qpc_path.get("lang_prefill_qpc_path"),
        kv_dma_share=True,
        full_batch_size=full_batch_size,
        cluster_id="prefill",
    )

    assert "image_idx" in prefill_session.binding_index_map, "image_idx not a compiled prefill input binding"
    if prefill_only:
        # The prefill handoff API requires host KV buffers for its final chunk.
        # Decode is deliberately absent in this mode, so use the equivalent
        # prefill-session buffer geometry instead.
        kv_caches = [np.zeros(shape, dtype=dtype) for (shape, dtype) in prefill_session.kv_cache_info]
        decode_session = None
        decode_has_image_idx = False
    else:
        decode_qpc_path = qeff_model.compile(**DECODE_CONFIG)
        decode_session = QAICInferenceSession(
            decode_qpc_path.get("lang_decode_qpc_path"),
            kv_dma_share=True,
            full_batch_size=full_batch_size,
            cluster_id="decode",
        )
        assert "batch_index" in decode_session.binding_index_map, "batch_index not a compiled decode input binding"
        decode_has_image_idx = "image_idx" in decode_session.binding_index_map
        # Shared host KV arrays, allocated once in decode-map order. Under CB the leading batch
        # dim is full_batch_size, so each family is [N, ...]: prefill writes one row, decode
        # reads/writes all N. Hybrid caches carry mixed 4-D (full) and 3-D (linear) shapes.
        kv_caches = [np.zeros(shape, dtype=dtype) for (shape, dtype) in decode_session.kv_cache_info]

    assert kv_caches and kv_caches[0].shape[0] == full_batch_size, (
        f"decode KV batch dim {kv_caches[0].shape[0] if kv_caches else None} != full_batch_size {full_batch_size}"
    )
    decode_kv_map = (
        decode_session.decode_buff_map + decode_session.decode_rs_kv_only_buff_map if decode_session is not None else []
    )

    def _prepare_prompt(prompt: str, image_url: str):
        """Tokenise + (optionally) run the vision QPC for one prompt"""
        if skip_vision:
            content = [{"type": "text", "text": prompt}]
        else:
            response = requests.get(image_url, stream=True)
            response.raise_for_status()
            image = Image.open(response.raw).convert("RGB")
            image = image.resize(
                (VISION_IMAGE_WIDTH, VISION_IMAGE_HEIGHT),
                Image.Resampling.LANCZOS,
            )
            content = [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info([messages])
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = qeff_model.model.prepare_inputs_for_generation(
            inputs=inputs, prefill_seq_len=prefill_seq_len, batch_size=1
        )

        pad_token_id = 1
        input_ids_length = inputs["input_ids"].shape[1]
        num_chunks = -(input_ids_length // -prefill_seq_len)  # ceil divide without float
        padded_len = num_chunks * prefill_seq_len

        inputs["input_ids"] = torch.nn.functional.pad(
            inputs["input_ids"], (0, padded_len - input_ids_length), "constant", pad_token_id
        )
        inputs["attention_mask"] = torch.nn.functional.pad(
            inputs["attention_mask"], (0, padded_len - input_ids_length), "constant", 0
        )
        for k, v in inputs.items():
            inputs[k] = np.array(v)

        vision_input_names = set(vision_session.input_names) if vision_session is not None else set()
        vision_inputs = {
            k: v for k, v in inputs.items() if k in VISION_INPUT_KEYS and k in vision_input_names
        }
        vision_inputs.update({k: vision_inputs[k].astype("float16") for k in VISION_FP16_KEYS if k in vision_inputs})
        vision_outputs = {}
        if not skip_vision and vision_inputs:
            raw = vision_session.run(vision_inputs)
            vision_outputs = {k: raw[k] for k in VISION_OUTPUT_KEYS if k in raw}

        lang_inputs = {k: v for k, v in inputs.items() if k not in vision_inputs}
        if "position_ids" in inputs:
            lang_inputs["position_ids"] = inputs["position_ids"]
            lang_inputs.pop("attention_mask", None)
        else:
            lang_inputs["position_ids"] = np.where(
                lang_inputs.pop("attention_mask"), np.arange(padded_len), -1
            )  # -1 marks invalid positions
        lang_inputs["image_idx"] = np.array([[0]])
        num_pos_sections = lang_inputs["position_ids"].shape[0]
        return lang_inputs, vision_outputs, num_chunks, num_pos_sections

    def _prefill_slot(lang_inputs, vision_outputs, num_chunks, slot: int):
        """Chunked prefill of one prompt into KV ``slot``.

        The prefill QPC is compiled for batch one. The target host KV row selects the
        decode slot, while the device batch index remains zero. Returns
        ``(first_token, phys_pos, mrope_pos)``.
        """
        chunk_inputs = dict(lang_inputs)
        chunk_inputs["batch_index"] = np.array([[0]], dtype=np.int64)
        if not skip_vision:
            for k in VISION_OUTPUT_KEYS:
                if k in vision_outputs:
                    chunk_inputs[k] = vision_outputs[k]
        slot_kv_view = [kv[slot : slot + 1] for kv in kv_caches]
        exec_idx = None
        start_timer = perf_counter()
        for i in range(num_chunks):
            chunk_inputs["input_ids"] = lang_inputs["input_ids"][:, i * prefill_seq_len : (i + 1) * prefill_seq_len]
            chunk_inputs["position_ids"] = lang_inputs["position_ids"][
                ..., i * prefill_seq_len : (i + 1) * prefill_seq_len
            ]
            last_chunk = i == num_chunks - 1
            exec_idx = prefill_session.np_run_pipeline(
                chunk_inputs,
                last_chunk=last_chunk,
                kv_cache_buffers=slot_kv_view if last_chunk else None,
            )
            prefill_session.complete_inf(exec_idx, is_prefill=True)
            chunk_inputs["image_idx"] = prefill_session.get_outputs(index=exec_idx)["image_idx_output"]
            if(i == 0):
                end_timer = perf_counter() - start_timer
                print(f"prefill chunk {i + 1}/{num_chunks} time={end_timer:.2f} secs")
        prefill_out = prefill_session.get_outputs(index=exec_idx)
        first_token = int(np.argmax(prefill_out["logits"]))
        phys_pos = int(lang_inputs["position_ids"][0].max()) + 1
        mrope_pos = (
            int(lang_inputs["position_ids"][1:].max()) + 1 if lang_inputs["position_ids"].shape[0] > 1 else phys_pos
        )
        return first_token, phys_pos, mrope_pos

    # Per-slot decode state.
    ongoing = [False] * full_batch_size
    last_token = [0] * full_batch_size
    phys_pos = [0] * full_batch_size
    mrope_pos = [0] * full_batch_size
    gen_count = [0] * full_batch_size
    slot_prompt_idx = [-1] * full_batch_size
    slot_tokens = [None] * full_batch_size
    results = [None] * len(prompts)
    num_pos_sections = 1
    vision_outputs_ref = None

    def _seed_slot(slot, prompt_idx, first_token, phys, mrope):
        slot_prompt_idx[slot] = prompt_idx
        slot_tokens[slot] = [first_token]
        gen_count[slot] = 1
        last_token[slot] = first_token
        phys_pos[slot] = phys
        mrope_pos[slot] = mrope
        ongoing[slot] = True

    prompt_queue = deque((idx, prompt, image_urls[idx % len(image_urls)]) for idx, prompt in enumerate(prompts))

    prefill_start = perf_counter()
    for slot in range(full_batch_size):
        if not prompt_queue:
            break
        prompt_idx, prompt, prompt_image_url = prompt_queue.popleft()
        lang_inputs, vision_outputs, num_chunks, num_pos_sections = _prepare_prompt(prompt, prompt_image_url)
        vision_outputs_ref = vision_outputs if vision_outputs else vision_outputs_ref
        ft, phys, mrope = _prefill_slot(lang_inputs, vision_outputs, num_chunks, slot)
        _seed_slot(slot, prompt_idx, ft, phys, mrope)
    print(f"Initial prefill time : {perf_counter() - prefill_start:.2f} secs")

    if prefill_only:
        first_tokens = [slot_tokens[slot] for slot in range(full_batch_size) if ongoing[slot]]
        print(f"Prefill-only first token IDs: {[tokens[0] for tokens in first_tokens]}")
        print("Stopped after initial prefill (no decode compile or generation).")
        return {"first_tokens": [tokens[0] for tokens in first_tokens]}

    if not skip_vision and vision_outputs_ref:
        decode_session.set_persistent_inputs(
            {k: np.zeros_like(v) for k, v in vision_outputs_ref.items() if k in decode_session.binding_index_map}
        )

    def _build_decode_inputs():
        input_ids = np.full((full_batch_size, 1), -1, dtype=np.int64)
        position_ids = np.full((num_pos_sections, full_batch_size, 1), -1, dtype=np.int64)
        batch_index = np.full((full_batch_size, 1), -1, dtype=np.int64)
        for slot in range(full_batch_size):
            if not ongoing[slot]:
                continue
            input_ids[slot, 0] = last_token[slot]
            position_ids[0, slot, 0] = phys_pos[slot]
            position_ids[1:, slot, 0] = mrope_pos[slot]
            batch_index[slot, 0] = slot
        decode_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "batch_index": batch_index,
        }
        if decode_has_image_idx:
            decode_inputs["image_idx"] = np.array([[0]], dtype=np.int64)
        return decode_inputs

    st = perf_counter()
    decode_steps = 0
    while any(ongoing):
        # Wire the full [N, ...] KV buffers once (identity: device row i <-> host row i);
        # per-slot addressing is carried by the decode batch_index input above.
        decode_session.set_data_for_kv_handoff(
            kv_caches + kv_caches,
            [("batch_index", 0), ("ctx_start", 0)],
            index=decode_session.decode_execObj_idx,
            buff_map=decode_kv_map,
        )
        decode_inputs = _build_decode_inputs()
        exec_idx = decode_session.np_run(decode_inputs, is_prefill=False)
        decode_session.complete_inf(exec_idx, is_prefill=False)
        out = decode_session.get_outputs(index=exec_idx)
        decode_steps += 1

        logits = out["logits"]
        logits = logits.reshape(full_batch_size, -1, logits.shape[-1])[:, -1, :]
        next_tokens = np.argmax(logits, axis=-1)

        for slot in range(full_batch_size):
            if not ongoing[slot]:
                continue
            tok = int(next_tokens[slot])
            if tok == tokenizer.eos_token_id or gen_count[slot] >= generation_len:
                results[slot_prompt_idx[slot]] = slot_tokens[slot]
                if prompt_queue:
                    prompt_idx, prompt, prompt_image_url = prompt_queue.popleft()
                    lang_inputs, vision_outputs, num_chunks, num_pos_sections = _prepare_prompt(
                        prompt, prompt_image_url
                    )
                    ft, phys, mrope = _prefill_slot(lang_inputs, vision_outputs, num_chunks, slot)
                    _seed_slot(slot, prompt_idx, ft, phys, mrope)
                else:
                    ongoing[slot] = False
            else:
                slot_tokens[slot].append(tok)
                gen_count[slot] += 1
                last_token[slot] = tok
                phys_pos[slot] += 1
                mrope_pos[slot] += 1
    ft = perf_counter()

    total_tokens = sum(len(t) for t in results if t)
    print(f"decode steps={decode_steps} tok/sec={total_tokens / (ft - st):.2f}")
    first_tokens = []
    for idx, prompt in enumerate(prompts):
        toks = results[idx] or []
        first_tokens.append(toks[0] if toks else None)
        print(f"\ninput [{idx}]\n{prompt}\noutput\n{tokenizer.decode(toks)}")

    return {"first_tokens": first_tokens, "tokens": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF model id")
    parser.add_argument("--prompt", action="append", dest="prompts", help="prompt (repeatable); defaults to a set of 4")
    parser.add_argument("--prefill-seq-len", type=int, default=DEFAULT_PREFILL_SEQ_LEN)
    parser.add_argument("--ctx-len", type=int, default=DEFAULT_CTX_LEN)
    parser.add_argument("--generation-len", type=int, default=DEFAULT_GENERATION_LEN)
    parser.add_argument("--full-batch-size", type=int, default=DEFAULT_FULL_BATCH_SIZE, help="CB decode width (N)")
    parser.add_argument("--stages", type=int, default=STAGES, help="prefill pipeline depth (mdp_num_partitions)")
    parser.add_argument(
        "--prefill-num-devices", type=int, default=PREFILL_NUM_DEVICES, help="num devices for the prefill QPC"
    )
    parser.add_argument(
        "--decode-num-devices", type=int, default=DECODE_NUM_DEVICES, help="num devices for the decode QPC"
    )
    parser.add_argument(
        "--mdp-load-partition-config",
        help="Path to a pre-generated MDP JSON, including an MDP regex JSON.",
    )
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        help="Test-only language-layer count; omit to use the full 30B model architecture.",
    )
    parser.add_argument(
        "--vision-depth",
        type=int,
        help="Test-only vision-layer count; updates deepstack_visual_indexes accordingly.",
    )
    parser.add_argument(
        "--generate-mdp-only",
        action="store_true",
        help="Write MDP artifacts and stop before final compile, decode compile, and inference.",
    )
    parser.add_argument(
        "--prefill-only",
        action="store_true",
        help="Compile and run only the initial prefill; skip decode compilation and generation.",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Compile prefill and decode QPCs, then exit before runtime session creation or inference.",
    )
    vision = parser.add_mutually_exclusive_group()
    vision.add_argument("--skip-vision", dest="skip_vision", action="store_true", help="text-only lang path")
    vision.add_argument(
        "--with-vision",
        dest="skip_vision",
        action="store_false",
        help="image+text: compile and run the vision QPC, pairing each prompt with an --image-url (default)",
    )
    parser.set_defaults(skip_vision=False)
    parser.add_argument(
        "--image-url",
        action="append",
        dest="image_urls",
        help="image URL (repeatable); paired with prompts by index. Defaults to a set of 4",
    )
    args = parser.parse_args()

    # With vision the default text prompts are a poor fit; use the image prompts unless the
    # user supplied their own --prompt.
    prompts = args.prompts
    image_urls = args.image_urls
    if not args.skip_vision:
        if not prompts:
            prompts = list(DEFAULT_IMAGE_PROMPTS)
        if not image_urls:
            image_urls = list(DEFAULT_IMAGE_URLS)

    run(
        model_id=args.model_id,
        prompts=prompts,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        generation_len=args.generation_len,
        full_batch_size=args.full_batch_size,
        skip_vision=args.skip_vision,
        image_urls=image_urls,
        stages=args.stages,
        prefill_num_devices=args.prefill_num_devices,
        decode_num_devices=args.decode_num_devices,
        mdp_load_partition_config=args.mdp_load_partition_config,
        num_hidden_layers=args.num_hidden_layers,
        vision_depth=args.vision_depth,
        generate_mdp_only=args.generate_mdp_only,
        prefill_only=args.prefill_only,
        compile_only=args.compile_only,
    )
