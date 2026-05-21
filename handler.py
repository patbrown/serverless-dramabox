import logging

import runpod

from dramabox_runtime import safe_generate


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


def handler(job):
    return safe_generate((job or {}).get("input", {}))


runpod.serverless.start({"handler": handler})
