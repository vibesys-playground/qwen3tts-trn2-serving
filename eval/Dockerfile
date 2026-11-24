# The task image (schema 0.2): the judge. It runs the check and the goodput sweep against the
# candidate's server at VIBESYS_SERVICE_URL; it has no Neuron device and no candidate code.
#
# The base is the vibesys-evaluator llm-serving image (Python 3.12, aiohttp 3.13.5, numpy 2.5.4,
# scipy 1.18.1, transformers 5.22.1) BUILT LOCALLY on this host (ghcr.io refuses anonymous pulls);
# vibesys-cli keys this image by the local image ID of the base, so a rebuild of the tag is
# noticed. CPU torch is added for the check's Whisper transcription.
FROM vibesys-evaluator-llm-serving:0.3.0rc3-local

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.15.0
COPY . /task
# The operator's Hugging Face cache, mounted read-only as "hf": Whisper is read from it offline.
ENV HF_HOME=/vibesys/mounts/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
RUN python -I -c "import aiohttp, numpy, scipy.signal, torch, transformers; print(torch.__version__, transformers.__version__)" \
 && python -I -c "import json; assert len(open('/task/data/libritts_test_clean.jsonl').read().splitlines()) == 4837"
