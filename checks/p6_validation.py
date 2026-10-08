"""P6 checks: interleaving order, download idempotency, VGGSound 30-clip audio study, demux vs ffmpeg."""
import glob
import logging
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

import fiftyone as fo
import fiftyone.zoo as foz

logging.basicConfig(level=logging.WARNING)
NAME = "google/embeddinggemma-2"
SR = 16000
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


model = foz.load_zoo_model(NAME, media_type="video", modalities=["vision", "audio"])
media = sys.modules[type(model).__module__.rsplit(".", 1)[0] + ".media"]
pkg = sys.modules[type(model).__module__.rsplit(".", 1)[0]]

# ------------------------------------------------------------ interleaving order
print("\n== interleaving")
imgs = sorted(glob.glob(os.path.expanduser("~/fiftyone/quickstart/data/*.jpg")))[:2]
a, b = [Image.open(p).convert("RGB") for p in imgs]
tok = model._processor.image_token
ab, ba, ab2 = model.embed_interleaved([
    {"text": f"first {tok} then {tok}", "images": [a, b]},
    {"text": f"first {tok} then {tok}", "images": [b, a]},
    {"text": f"first {tok} then {tok}", "images": [a, b]},
])
check("swapping two images changes the vector", float(ab @ ba) < 0.9999, f"cos {float(ab @ ba):.4f}")
check("same order is deterministic", float(ab @ ab2) > 0.99999, f"cos {float(ab @ ab2):.6f}")
txt_img = model.embed_interleaved([{"text": f"a photo {tok}", "images": [a]}])[0]
img_only = model.embed_images([a])[0]
check("text+image differs from image alone", float(txt_img @ img_only) < 0.9999, f"cos {float(txt_img @ img_only):.4f}")

# ------------------------------------------------------------ download_model idempotency
print("\n== download_model")
path = os.path.join(os.path.dirname(pkg.__file__), "embeddinggemma-2")
before = {f: os.path.getmtime(os.path.join(path, f)) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))}
t0 = time.time()
pkg.download_model(NAME, path)
pkg.download_model(NAME, path)
after = {f: os.path.getmtime(os.path.join(path, f)) for f in before}
check("download_model twice on existing dir is a no-op", before == after, f"{time.time() - t0:.1f}s for 2 calls")

# ------------------------------------------------------------ demux: PyAV vs ffmpeg CLI
print("\n== demux vs ffmpeg CLI")
vg_all = fo.load_dataset("VGGSound")
classes = sorted(vg_all.distinct("ground_truth.label"))
view = vg_all.match_tags("test").take(30, seed=51)
paths, labels = view.values(["filepath", "ground_truth.label"])


def ffmpeg_audio(p):
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32)


corrs, coss = [], []
for p in paths[:5]:
    ours = media.read_audio(p)
    ref = ffmpeg_audio(p)
    n = min(len(ours), len(ref))
    corrs.append(float(np.corrcoef(ours[:n], ref[:n])[0, 1]))
    e = model.embed_audio([ours, ref])
    coss.append(float(e[0] @ e[1]))
check("PyAV audio == ffmpeg CLI audio (corr > 0.99)", min(corrs) > 0.99, np.round(corrs, 4))
check("PyAV vs ffmpeg CLI audio embeddings (cos > 0.999)", min(coss) > 0.999, np.round(coss, 5))

waves = [media.read_audio(p) for p in paths]
rms = np.array([np.sqrt(np.mean(w ** 2)) for w in waves])
peak = np.array([np.abs(w).max() for w in waves])
dur = np.array([len(w) / SR for w in waves])
print(f"    30 clips: duration {dur.min():.1f}-{dur.max():.1f}s | RMS median {np.median(rms):.3f} "
      f"(min {rms.min():.4f}) | peak median {np.median(peak):.2f} | silent (<0.001 RMS): {(rms < 1e-3).sum()}")
check("no silent or empty audio among 30 clips", (rms > 1e-3).all() and (dur > 1).all())

# ------------------------------------------------------------ 30-clip study
print(f"\n== VGGSound, 30 test clips, {len(set(labels))} distinct classes, zero-shot over {len(classes)}")
t0 = time.time()
ev = np.stack(model.embed_dataset(view, modality="video"))
ea = np.stack(model.embed_dataset(view, modality="audio"))
eva = np.stack(model.embed_dataset(view, modality="video+audio"))
print(f"    embedded 30 clips x 3 modalities in {time.time() - t0:.1f}s")
truth = np.array([classes.index(l) for l in labels])


def topk(E, C):
    S = E @ C.T
    ranks = (S > S[np.arange(len(E)), truth][:, None]).sum(1) + 1
    return (ranks == 1).mean(), (ranks <= 5).mean(), int(np.median(ranks))


C_default = model.embed_text(classes, prompt_name="SearchQuery")
rows = {}
for name, E in [("video", ev), ("audio", ea), ("video+audio", eva)]:
    rows[name] = topk(E, C_default)
    print(f"    {name:12s} top-1 {rows[name][0]:.2f}  top-5 {rows[name][1]:.2f}  median rank {rows[name][2]}")

print("\n    audio-only prompt grid (top-1 / top-5 / median rank):")
best = None
for prefix in ["SearchQuery", "Classification", None]:
    for template in ["{}", "the sound of {}", "a recording of {}"]:
        C = model.embed_text([template.format(c) for c in classes], prompt_name=prefix)
        r = topk(ea, C)
        print(f"      {str(prefix):14s} {template!r:22s} {r[0]:.2f} / {r[1]:.2f} / {r[2]}")
        if best is None or r[1] > best[1][1]:
            best = ((prefix, template), r)
print(f"    best audio-only setting: {best[0]} -> top-1 {best[1][0]:.2f}, top-5 {best[1][1]:.2f}")

check("video top-5 >= 0.6 over 309 classes", rows["video"][1] >= 0.6, f"{rows['video'][1]:.2f}")
check("video+audio not worse than video at top-5", rows["video+audio"][1] >= rows["video"][1] - 0.05,
      f"{rows['video+audio'][1]:.2f} vs {rows['video'][1]:.2f}")

print(f"\n{sum(results)}/{len(results)} passed")
