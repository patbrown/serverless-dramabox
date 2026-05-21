import base64
import json
import logging
import os
import time
import traceback
import uuid
from pathlib import Path

import torch

from src.inference_server import TTSServer
from src.model_downloader import get_all_paths


LOGGER = logging.getLogger("dramabox-runtime")
LOGGER.setLevel(logging.INFO)

WORKER_VERSION = "dramabox-serverless-2026-05-21-v3-no-compile-default"

_SERVER = None
_SERVER_LOAD_SECONDS = None
_MODEL_PATHS = None
_SERVER_OPTIONS = None


def _now_ms():
    return int(time.time() * 1000)


def _path_from_env(name, default):
    value = os.environ.get(name)
    if value:
        return Path(value)
    return Path(default)


def cache_dir():
    if os.environ.get("DRAMABOX_CACHE_DIR"):
        return Path(os.environ["DRAMABOX_CACHE_DIR"])
    if Path("/runpod-volume").exists():
        return Path("/runpod-volume/dramabox-cache")
    if Path("/workspace").exists():
        return Path("/workspace/dramabox-cache")
    return Path("/tmp/dramabox-cache")


def output_dir():
    path = _path_from_env("DRAMABOX_OUTPUT_DIR", "/tmp/dramabox-output")
    path.mkdir(parents=True, exist_ok=True)
    return path


def gpu_snapshot(label):
    if not torch.cuda.is_available():
        return {"label": label, "cuda_available": False}

    torch.cuda.synchronize()
    props = torch.cuda.get_device_properties(0)
    return {
        "label": label,
        "cuda_available": True,
        "device": torch.cuda.get_device_name(0),
        "total_vram_gb": round(props.total_memory / 1024**3, 2),
        "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 3),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "max_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
    }


def _coerce_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _normalize_input(input_payload):
    payload = dict(input_payload or {})
    prompt = payload.get("prompt") or payload.get("text")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Dramabox request requires non-empty input.prompt or input.text.")

    normalized = {
        "prompt": prompt.strip(),
        "voice_ref": payload.get("voice_ref"),
        "cfg_scale": float(payload.get("cfg_scale", 2.5)),
        "stg_scale": float(payload.get("stg_scale", 1.5)),
        "duration_multiplier": float(payload.get("duration_multiplier", 1.1)),
        "seed": int(payload.get("seed", 42)),
        "watermark": _coerce_bool(payload.get("watermark"), True),
    }
    return normalized


def get_server():
    global _SERVER
    global _SERVER_LOAD_SECONDS
    global _MODEL_PATHS
    global _SERVER_OPTIONS

    if _SERVER is not None:
        return _SERVER

    selected_cache = cache_dir()
    selected_cache.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(selected_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(selected_cache))

    LOGGER.info("Dramabox cache dir: %s", selected_cache)
    started = time.time()
    _MODEL_PATHS = get_all_paths(str(selected_cache))
    LOGGER.info("Dramabox model paths: %s", json.dumps(_MODEL_PATHS, sort_keys=True))

    _SERVER_OPTIONS = {
        "compile_model": _coerce_bool(os.environ.get("DRAMABOX_COMPILE_MODEL"), False),
        "bnb_4bit": _coerce_bool(os.environ.get("DRAMABOX_BNB_4BIT"), True),
        "dtype": os.environ.get("DRAMABOX_DTYPE", "bf16"),
    }
    LOGGER.info("Dramabox server options: %s", json.dumps(_SERVER_OPTIONS, sort_keys=True))

    _SERVER = TTSServer(
        checkpoint=_MODEL_PATHS["transformer"],
        full_checkpoint=_MODEL_PATHS["audio_components"],
        gemma_root=_MODEL_PATHS["gemma_root"],
        device="cuda",
        dtype=_SERVER_OPTIONS["dtype"],
        compile_model=_SERVER_OPTIONS["compile_model"],
        bnb_4bit=_SERVER_OPTIONS["bnb_4bit"],
    )
    _SERVER_LOAD_SECONDS = round(time.time() - started, 3)
    LOGGER.info("Dramabox server loaded in %.3fs", _SERVER_LOAD_SECONDS)
    return _SERVER


def health():
    loaded = _SERVER is not None
    return {
        "status": "ok",
        "worker_version": WORKER_VERSION,
        "loaded": loaded,
        "model_load_seconds": _SERVER_LOAD_SECONDS,
        "server_options": _SERVER_OPTIONS,
        "cache_dir": str(cache_dir()),
        "output_dir": str(output_dir()),
        "gpu": gpu_snapshot("health"),
    }


def generate(input_payload):
    started_ms = _now_ms()
    stage = "normalize_input"
    request = _normalize_input(input_payload)

    stage = "get_server"
    server = get_server()

    output_path = output_dir() / f"dramabox-{uuid.uuid4().hex}.wav"

    generate_kwargs = {
        "prompt": request["prompt"],
        "output": str(output_path),
        "cfg_scale": request["cfg_scale"],
        "stg_scale": request["stg_scale"],
        "duration_multiplier": request["duration_multiplier"],
        "seed": request["seed"],
        "watermark": request["watermark"],
    }
    if request["voice_ref"]:
        generate_kwargs["voice_ref"] = request["voice_ref"]

    before_gpu = gpu_snapshot("before_generate")
    gen_started = time.time()
    stage = "generate_to_file"
    result = server.generate_to_file(**generate_kwargs)
    generation_seconds = round(time.time() - gen_started, 3)

    stage = "after_generate_gpu_snapshot"
    after_gpu = gpu_snapshot("after_generate")

    stage = "read_output"
    if not output_path.exists():
        raise FileNotFoundError(f"Dramabox did not create output file: {output_path}")

    artifact_bytes = output_path.read_bytes()

    stage = "base64_encode_output"
    artifact_base64 = base64.b64encode(artifact_bytes).decode("ascii")

    return {
        "status": "COMPLETED",
        "worker_version": WORKER_VERSION,
        "output": {
            "artifact_base64": artifact_base64,
            "artifact_filename": output_path.name,
            "artifact_path": str(output_path),
            "content_type": "audio/wav",
            "output_bytes": len(artifact_bytes),
            "result": str(result),
            "generation_seconds": generation_seconds,
            "model_load_seconds": _SERVER_LOAD_SECONDS,
        },
        "request": request,
        "timing": {
            "started_ms": started_ms,
            "ended_ms": _now_ms(),
            "total_seconds": round((_now_ms() - started_ms) / 1000.0, 3),
        },
        "stage": stage,
        "gpu": {
            "before": before_gpu,
            "after": after_gpu,
        },
    }


def safe_generate(input_payload):
    try:
        if isinstance(input_payload, dict) and input_payload.get("health"):
            return health()
        return generate(input_payload)
    except Exception as exc:
        LOGGER.exception("Dramabox generation failed.")
        return {
            "status": "FAILED",
            "worker_version": WORKER_VERSION,
            "failure_class": exc.__class__.__name__,
            "failure_message": str(exc),
            "failure_traceback": traceback.format_exc(),
            "gpu": gpu_snapshot("failure"),
        }


if os.environ.get("DRAMABOX_PRELOAD", "1") != "0":
    try:
        get_server()
    except Exception:
        LOGGER.exception("Dramabox preload failed.")
        raise
