"""P4 checks: video leaf through FiftyOne (full videos, clips, VGGSound, VidMuse)."""
import logging
import time

import numpy as np

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz

logging.basicConfig(level=logging.WARNING)
NAME = "google/embeddinggemma-2"
results = []
clones = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def clone(view, name):
    if name in fo.list_datasets():
        fo.delete_dataset(name)
    clones.append(name)
    return view.clone(name)


def unit(e):
    return np.abs(np.linalg.norm(e, axis=-1) - 1).max() < 1e-5


try:
    model = foz.load_zoo_model(NAME, media_type="video")
    check("load media_type=video", type(model).__name__ == "EmbeddingGemma2VideoModel" and model.media_type == "video")
    weights = id(model._model)

    # ------------------------------------------------------------ quickstart-video
    qv = clone(fo.load_dataset("quickstart-video"), "eg2-p4-qv")
    t0 = time.time()
    qv.compute_embeddings(model, embeddings_field="eg2")
    emb = np.stack(qv.values("eg2"))
    check("quickstart-video: one vector per video", emb.shape == (len(qv), 768) and unit(emb),
          f"{emb.shape} in {time.time() - t0:.1f}s")
    check("quickstart-video: embed(filepath) == compute_embeddings",
          float(model.embed(qv.first().filepath) @ emb[0]) > 0.9999)

    # clips: a 2 s window of the first video
    s = qv.first()
    fr = s.metadata.frame_rate
    s["eg2_seg"] = fo.TemporalDetections(detections=[fo.TemporalDetection(label="seg", support=[1, int(2 * fr)])])
    s.save()
    clips = qv.to_clips("eg2_seg")
    clip_emb = clips.compute_embeddings(model)
    direct = model.embed_frames([model.load_frames(s.filepath, start=0, end=int(2 * fr) / fr)])[0]
    check("clip: compute_embeddings on clips view == direct window embed",
          float(clip_emb[0] @ direct) > 0.9999, f"{float(clip_emb[0] @ direct):.5f}")
    check("clip: differs from full-video vector", float(clip_emb[0] @ emb[0]) < 0.999,
          f"{float(clip_emb[0] @ emb[0]):.4f}")

    # ------------------------------------------------------------ VGGSound
    vg_all = fo.load_dataset("VGGSound")
    classes = sorted(vg_all.distinct("ground_truth.label"))
    vg = clone(vg_all.take(3, seed=51), "eg2-p4-vgg")
    vg.compute_embeddings(model, embeddings_field="eg2")
    check("VGGSound: 3 vectors written", len(vg.exists("eg2")) == 3 and unit(np.stack(vg.values("eg2"))))

    model.classes = classes
    vg.apply_model(model, label_field="eg2_zs")
    gt, pred = vg.values("ground_truth.label"), vg.values("eg2_zs.label")
    check("VGGSound zero-shot over 309 classes (spike: 3/3)", sum(a == b for a, b in zip(gt, pred)) >= 2,
          list(zip(gt, pred, [round(c, 3) for c in vg.values("eg2_zs.confidence")])))

    fob.compute_similarity(vg, embeddings="eg2", model=NAME, model_kwargs={"media_type": "video"},
                           brain_key="eg2_sim")
    idx = vg.load_brain_results("eg2_sim")
    check("VGGSound: index reloads with prompt support", idx.config.supports_prompts)
    hits = 0
    for label in gt:
        top = vg.sort_by_similarity(label, k=1, brain_key="eg2_sim").first()
        hits += top.ground_truth.label == label
    check("VGGSound: text query of each true label ranks its clip first", hits >= 2, f"{hits}/3")

    # ------------------------------------------------------------ VidMuse video slice
    vm = clone(fo.load_dataset("VidMuse-V2M").select_group_slices("video").take(3, seed=51), "eg2-p4-vm")
    t0 = time.time()
    vm.compute_embeddings(model, embeddings_field="eg2")
    check("VidMuse video slice: 3 vectors written", len(vm.exists("eg2")) == 3,
          f"{time.time() - t0:.1f}s for 3 long videos")
    fob.compute_similarity(vm, embeddings="eg2", model=NAME, model_kwargs={"media_type": "video"},
                           brain_key="eg2_sim")
    view = vm.sort_by_similarity("a music video with a singer", k=3, brain_key="eg2_sim")
    check("VidMuse: text query runs", len(view) == 3)

    # ------------------------------------------------------------ setters + budget
    p = vm.first().filepath
    base = model.embed(p)
    model.fps = 0.25
    slow = model.embed(p)
    model.fps = 1
    check("fps setter changes vector", float(base @ slow) < 0.9999, f"{float(base @ slow):.4f}")
    model.max_frames = 100
    n = len(model.load_frames(p))
    big = model.embed(p)
    model.max_frames = 32
    check("max_frames=100 fitted to token budget", n == 57 and big.shape == (768,), f"{n} frames")
    check("setters keep weights", id(model._model) == weights)

    # image model on a video collection: per-frame fallback must warn
    class _Capture(logging.Handler):
        def __init__(self):
            super().__init__()
            self.messages = []

        def emit(self, record):
            self.messages.append(record.getMessage())

    cap = _Capture()
    logging.getLogger("embeddinggemma2.zoo").addHandler(cap)
    img_model = foz.load_zoo_model(NAME)
    one = clone(fo.load_dataset("quickstart-video").limit(1), "eg2-p4-warn")
    one.compute_embeddings(img_model, embeddings_field="eg2_frames", batch_size=32)
    warned = any("Image model applied to a video collection" in m for m in cap.messages)
    check("image model on video dataset warns about per-frame fallback",
          warned and one.has_frame_field("eg2_frames") and img_model.mode is None)
finally:
    for name in clones:
        if name in fo.list_datasets():
            fo.delete_dataset(name)

print(f"\n{sum(results)}/{len(results)} passed")
