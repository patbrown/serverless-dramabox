import logging

import runpod

from dramabox_runtime import safe_generate


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


def handler(job):
    job = job or {}
    input_payload = dict(job.get("input", {}) or {})
    if job.get("id"):
        input_payload["_runpod_job_id"] = job["id"]
    return safe_generate(input_payload)


runpod.serverless.start({"handler": handler})
