"""P5 checks: audio leaf, embed_dataset / classify_dataset, video+audio, MCAP."""
import glob
import logging
import os
import time

import numpy as np

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz
from fiftyone import ViewField as F

logging.basicConfig(level=logging.WARNING)
NAME = "google/embeddinggemma-2"
results = []
clones = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def fresh(name):
    if name in fo.list_datasets():
        fo.delete_dataset(name)
    clones.append(name)
    return name


def unit(vs):
    vs = np.stack([v for v in vs if v is not None])
    return np.abs(np.linalg.norm(vs, axis=1) - 1).max() < 1e-5


try:
    audio = foz.load_zoo_model(NAME, media_type="audio")
    check("load media_type=audio (no vision encoder)",
          type(audio).__name__ == "EmbeddingGemma2AudioModel" and audio.config.modalities == ["audio"])
    av_model = foz.load_zoo_model(NAME, media_type="video", modalities=["vision", "audio"])

    # ------------------------------------------------------------ ESC-50 (36 clips, 6 classes)
    print("\n== ESC-50")
    cats = ["dog", "rain", "clock_tick", "siren", "keyboard_typing", "rooster"]
    nice = {"clock_tick": "a clock ticking", "keyboard_typing": "typing on a keyboard", "dog": "a dog barking",
            "rain": "rain falling", "siren": "a siren wailing", "rooster": "a rooster crowing"}
    esc = fo.Dataset(fresh("eg2-p5-esc"))
    esc.add_samples([fo.Sample(filepath=p, ground_truth=fo.Classification(label=os.path.basename(p).rsplit("_", 1)[0]))
                     for p in sorted(glob.glob("/tmp/esc50_wav/*.wav"))])
    t0 = time.time()
    audio.embed_dataset(esc, "eg2")
    check("ESC-50: embed_dataset writes 36 unit vectors", len(esc.exists("eg2")) == 36 and unit(esc.values("eg2")),
          f"{time.time() - t0:.1f}s, media_type={esc.media_type}")
    fob.compute_similarity(esc, embeddings="eg2", model=NAME, model_kwargs={"media_type": "audio"}, brain_key="eg2_sim")
    p5 = {c: np.mean([s.ground_truth.label == c for s in esc.sort_by_similarity(nice[c], k=5, brain_key="eg2_sim")])
          for c in cats}
    check("ESC-50: text->audio P@5 >= 0.8 via reloaded index (spike 0.83)", np.mean(list(p5.values())) >= 0.8,
          f"{np.mean(list(p5.values())):.2f} {p5}")
    audio.classes = [nice[c] for c in cats]
    audio.classify_dataset(esc, "eg2_zs", embeddings_field="eg2")
    inv = {v: k for k, v in nice.items()}
    acc = np.mean([inv[p] == g for p, g in zip(esc.values("eg2_zs.label"), esc.values("ground_truth.label"))])
    check("ESC-50: classify_dataset zero-shot accuracy (spike 0.75)", acc >= 0.7, f"{acc:.2f}")
    fob.compute_visualization(esc, embeddings="eg2", method="pca", brain_key="eg2_viz")
    check("ESC-50: compute_visualization runs", "eg2_viz" in esc.list_brain_runs())

    # ------------------------------------------------------------ VGGSound: audio and video+audio
    print("\n== VGGSound")
    vg_all = fo.load_dataset("VGGSound")
    classes = sorted(vg_all.distinct("ground_truth.label"))
    vg = vg_all.take(3, seed=51).clone(fresh("eg2-p5-vgg"))
    audio.embed_dataset(vg, "eg2_audio")
    av_model.embed_dataset(vg, "eg2_audio_full", modality="audio")
    a1, a2 = np.stack(vg.values("eg2_audio")), np.stack(vg.values("eg2_audio_full"))
    check("VGGSound: audio-only model == full model on audio", (a1 * a2).sum(1).min() > 0.999,
          np.round((a1 * a2).sum(1), 5))
    t0 = time.time()
    av_model.embed_dataset(vg, "eg2_va", modality="video+audio")
    check("VGGSound: video+audio writes 3 vectors", len(vg.exists("eg2_va")) == 3 and unit(vg.values("eg2_va")),
          f"{time.time() - t0:.1f}s")
    av_model.classes = classes
    av_model.classify_dataset(vg, "zs_va", embeddings_field="eg2_va")
    gt = vg.values("ground_truth.label")
    va_hits = sum(a == b for a, b in zip(gt, vg.values("zs_va.label")))
    check("VGGSound: video+audio zero-shot over 309 (spike 3/3)", va_hits >= 2,
          list(zip(gt, vg.values("zs_va.label"))))
    audio.classes = classes
    audio.classify_dataset(vg, "zs_audio", embeddings_field="eg2_audio")
    print("    audio-only zero-shot (spike: weak, ranks 94/41/295):", list(zip(gt, vg.values("zs_audio.label"))))

    # ------------------------------------------------------------ VidMuse: grouped clone, music slice
    print("\n== VidMuse-V2M (grouped)")
    vm_src = fo.load_dataset("VidMuse-V2M")
    ids = vm_src.select_group_slices("video").take(3, seed=51).values("youtube_id")
    vm = vm_src.match(F("youtube_id").is_in(ids)).clone(fresh("eg2-p5-vm"))
    try:
        audio.embed_dataset(vm, "eg2_audio")
        check("group dataset without slice raises", False)
    except ValueError as e:
        check("group dataset without slice raises", "select_group_slices" in str(e))
    music = vm.select_group_slices("music")
    audio.embed_dataset(music, "eg2_audio")
    check("VidMuse music slice: 3 vectors", len(music.exists("eg2_audio")) == 3, f"media_type={music.media_type}")
    video_view = vm.select_group_slices("video")
    av_model.embed_dataset(video_view, "eg2_video", modality="video")
    audio.embed_dataset(video_view, "eg2_mp4_audio", skip_failures=True)
    check("VidMuse mp4 without audio track: failure logged, None stored",
          len(video_view.exists("eg2_mp4_audio")) == 0 and len(video_view) == 3)
    fob.compute_similarity(music, embeddings="eg2_audio", model=NAME, model_kwargs={"media_type": "audio"},
                           brain_key="eg2_music_sim")
    ranks = []
    for s in video_view:
        order = music.sort_by_similarity(s.eg2_video, brain_key="eg2_music_sim").values("youtube_id")
        ranks.append(order.index(s.youtube_id) + 1)
    check("VidMuse: video->music cross-modal query runs on slice index", len(ranks) == 3,
          f"rank of paired music: {ranks} (n=3, spike diag won 1/3)")
    view = music.sort_by_similarity("upbeat electronic dance music", k=3, brain_key="eg2_music_sim")
    check("VidMuse: text query on music slice", len(view) == 3)

    long_wav = music.first().filepath
    audio.long_media = "chunk_mean"
    chunked = audio.embed(long_wav)
    audio.long_media = "truncate"
    trunc = audio.embed(long_wav)
    check("chunk_mean vs truncate differ on long music", float(chunked @ trunc) < 0.9999,
          f"cos {float(chunked @ trunc):.4f}")

    # ------------------------------------------------------------ VidMuse MCAP (multimodal)
    print("\n== VidMuse-V2M-MCAP")
    mc = fo.load_dataset("VidMuse-V2M-MCAP").match(F("youtube_id").is_in(ids)).clone(fresh("eg2-p5-mcap"))
    t0 = time.time()
    av_model.embed_dataset(mc, "eg2_video", modality="video")
    audio.embed_dataset(mc, "eg2_audio")
    av_model.embed_dataset(mc, "eg2_va", modality="video+audio")
    ok = all(len(mc.exists(f)) == 3 for f in ("eg2_video", "eg2_audio", "eg2_va"))
    check("MCAP: video, audio, video+audio fields written", ok, f"{time.time() - t0:.1f}s, media_type={mc.media_type}")
    mv = dict(zip(*video_view.values(["youtube_id", "eg2_video"])))
    mm = dict(zip(*music.values(["youtube_id", "eg2_audio"])))
    vc = [float(np.dot(v, mv[y])) for y, v in zip(*mc.values(["youtube_id", "eg2_video"]))]
    ac = [float(np.dot(v, mm[y])) for y, v in zip(*mc.values(["youtube_id", "eg2_audio"]))]
    check("MCAP vs mp4 video parity (>=0.99)", min(vc) >= 0.99, np.round(vc, 4))
    check("MCAP vs wav audio parity (>=0.999)", min(ac) >= 0.999, np.round(ac, 5))
    fob.compute_similarity(mc, embeddings="eg2_va", model=NAME,
                           model_kwargs={"media_type": "video", "modalities": ["vision", "audio"]},
                           brain_key="eg2_va_sim")
    view = mc.sort_by_similarity("a music video", k=3, brain_key="eg2_va_sim")
    check("MCAP: text query on multimodal index", len(view) == 3)

    # ------------------------------------------------------------ budget policy
    av_model.max_audio_seconds = 300
    f, s = av_model.fit_budget(num_frames=32, audio_seconds=300, text_tokens=1)
    av_model.max_audio_seconds = 30
    check("budget: video+audio keeps 32 frames, audio gets the rest", f == 32 and 0 < s < 300 and
          av_model._budget_tokens(f, s, 1) <= 8192, f"{f} frames, {s:.1f}s audio")
finally:
    for name in clones:
        if name in fo.list_datasets():
            fo.delete_dataset(name)

print(f"\n{sum(results)}/{len(results)} passed")
