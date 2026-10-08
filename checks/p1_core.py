"""P1 checks: zoo load, parity vs sentence-transformers, norms, setters, budget."""
import glob
import logging
import os
import time

import numpy as np
import torch
from PIL import Image

import fiftyone.zoo as foz

logging.basicConfig(level=logging.WARNING)
NAME = "google/embeddinggemma-2"
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def cos(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return (a * b).sum(-1)


t0 = time.time()
model = foz.load_zoo_model(NAME, modalities=["vision", "audio"])
print(f"load_zoo_model: {time.time() - t0:.1f}s | {type(model).__name__} | "
      f"{model.device} {model.dtype} | media_type={model.media_type}")
check("load via zoo", model.media_type == "image")

imgs = sorted(glob.glob(os.path.expanduser("~/fiftyone/quickstart/data/*.jpg")))[:4]
pil = [Image.open(p).convert("RGB") for p in imgs]
texts = ["a dog barking", "rain falling on a roof", "a red car"]
sr = 16000
tone = np.sin(2 * np.pi * 440 * np.arange(sr * 3) / sr).astype(np.float32)

ours_text = model.embed_text(texts, prompt_name="SearchQuery")
ours_prompts = model.embed_prompts(texts)
ours_img = model.embed_all(pil)
ours_aud = model.embed_audio([tone])

from sentence_transformers import SentenceTransformer

st = SentenceTransformer(NAME, model_kwargs={"torch_dtype": torch.bfloat16})
st_text = st.encode(texts, prompt_name="SearchQuery")
st_img = st.encode(pil)
st_aud = st.encode({"audio": tone})

c = cos(ours_text, st_text)
check("text parity vs ST (>=0.999)", c.min() >= 0.999, c.round(5))
check("embed_prompts == embed_text(SearchQuery)", np.allclose(ours_prompts, ours_text))
c = cos(ours_img, st_img)
check("image parity vs ST (>=0.999)", c.min() >= 0.999, c.round(5))
c = cos(ours_aud, st_aud[None])
check("audio parity vs ST (>=0.999)", c.min() >= 0.999, c.round(5))
del st
torch.cuda.empty_cache()

allv = np.concatenate([ours_text, ours_img, ours_aud])
norms = np.linalg.norm(allv, axis=1)
check("no NaN", not np.isnan(allv).any())
check("norms 1 +- 1e-5", np.abs(norms - 1).max() < 1e-5, f"max dev {np.abs(norms - 1).max():.2e}")
check("dtype float32", allv.dtype == np.float32)

weights = id(model._model)
for d in (512, 256, 128):
    model.embedding_dim = d
    e = model.embed_all(pil[:1])
    check(f"MRL {d}: shape + unit norm", e.shape == (1, d) and abs(np.linalg.norm(e) - 1) < 1e-5)
model.embedding_dim = 768

model.image_max_soft_tokens = 1120
hi = model.embed_all(pil[:1])
model.image_max_soft_tokens = 280
check("image_max_soft_tokens changes vector", cos(hi, ours_img[:1])[0] < 0.999, cos(hi, ours_img[:1]).round(4))
check("setters keep weights (identity)", id(model._model) == weights)

for bad in [("embedding_dim", 300), ("image_max_soft_tokens", 300), ("prompt_name", "Nope"), ("text_prompt", "no braces")]:
    try:
        setattr(model, *bad)
        check(f"setter rejects {bad}", False)
    except (ValueError, TypeError):
        check(f"setter rejects {bad}", True)

try:
    foz.load_zoo_model(NAME, dtype="float16")
    check("fp16 raises", False)
except ValueError:
    check("fp16 raises", True)

model.classes = ["cat", "dog", "car"]
model.text_prompt = "a photo of a {}"
labels = model.predict_all(pil)
check("zero-shot returns Classification per image", len(labels) == 4 and all(l.label in model.classes for l in labels),
      [(l.label, round(l.confidence, 3)) for l in labels])

# audio cap passes through to the feature extractor
long_wave = np.random.RandomState(0).randn(sr * 60).astype(np.float32) * 0.1
model.max_audio_seconds = 30
a30 = model.embed_audio([long_wave])
model.max_audio_seconds = 60
a60 = model.embed_audio([long_wave])
enc = model._processor(text=[model._processor.audio_token], audio=[[long_wave]], return_tensors="pt",
                       audio_kwargs=model._audio_kwargs())
n_aud = int((enc["input_ids"] == model._processor.audio_token_id).sum())
check("max_audio_seconds=60 gives 1500 audio tokens", n_aud == 1500, n_aud)
check("30s vs 60s cap changes vector", cos(a30, a60)[0] < 0.9999, cos(a30, a60).round(4))
model.max_audio_seconds = 30

# token budget
f, s = model.fit_budget(num_frames=100)
check("budget: 100 frames reduced to fit", f * 142 + 2 <= 8192 and f < 100, f)
model.max_audio_seconds = 400
f, s = model.fit_budget(num_frames=32, audio_seconds=400)
check("budget: 32 frames + 400s audio fits", model._budget_tokens(f, s) <= 8192, (f, round(s, 1)))
model.max_audio_seconds = 30

# selective loading
del model
torch.cuda.empty_cache()
img_only = foz.load_zoo_model(NAME)
n_img = sum(p.numel() for p in img_only._model.parameters()) / 1e6
check("default image load drops audio (~439M)", 430 < n_img < 445, f"{n_img:.1f}M")
try:
    img_only.embed_audio([tone])
    check("missing encoder raises", False)
except ValueError:
    check("missing encoder raises", True)
del img_only
torch.cuda.empty_cache()

aud_only = foz.load_zoo_model(NAME, modalities=["audio"])
n_aud_m = sum(p.numel() for p in aud_only._model.parameters()) / 1e6
e = aud_only.embed_audio([tone])
check("audio-only load (vision_config=None) works", e.shape == (1, 768), f"{n_aud_m:.1f}M params")
check("audio-only parity vs full model", cos(e, ours_aud)[0] >= 0.999, cos(e, ours_aud).round(5))

print(f"\n{sum(ok for _, ok in results)}/{len(results)} passed")
