"""P2 checks: media.py decode paths on VGGSound, VidMuse-V2M and VidMuse-V2M-MCAP."""
import importlib.util
import logging
import os
import time

import numpy as np

import fiftyone as fo
import fiftyone.zoo as foz

logging.basicConfig(level=logging.WARNING)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("eg2_media", os.path.join(ROOT, "media.py"))
media = importlib.util.module_from_spec(spec)
spec.loader.exec_module(media)

SR = media.AUDIO_SAMPLE_RATE
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def timed(fn):
    t = time.time()
    out = fn()
    return out, time.time() - t


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def frame_checks(tag, frames, times, duration, fps, max_frames, native_fps):
    expected = int(min(max_frames, max(1, int(duration * fps))))
    targets = media.sample_times(duration, fps, max_frames)
    lag = targets - np.array(times)
    check(f"{tag}: frames == requested", len(frames) == expected, f"{len(frames)}/{expected}")
    native_fps = native_fps or 30
    check(f"{tag}: frame at or before target, within 1 native frame",
          lag.min() >= -1e-3 and lag.max() <= 1.0 / native_fps + 1e-3,
          f"lag {lag.min() * 1000:.1f}..{lag.max() * 1000:.1f} ms")
    check(f"{tag}: times non-decreasing", all(np.diff(times) >= 0))
    distinct_times = len(set(times))
    distinct_imgs = len({np.asarray(f).tobytes().__hash__() for f in frames})
    check(f"{tag}: distinct timestamps give distinct frame objects",
          len({id(f) for f in frames}) == distinct_times, f"{distinct_imgs} distinct images / {distinct_times} times")


# ----------------------------------------------------------------- VGGSound
print("\n== VGGSound")
vg = fo.load_dataset("VGGSound").take(3, seed=51)
for s in vg:
    tag = f"VGGSound {s.youtube_id if s.has_field('youtube_id') else s.id}"
    (frames, times, dur), t = timed(lambda: media.read_video_frames(s.filepath, fps=1, max_frames=32))
    print(f"  {tag}: {dur:.2f}s, {len(frames)} frames in {t:.2f}s")
    frame_checks(tag, frames, times, dur, 1, 32, s.metadata.frame_rate)
    wave, t = timed(lambda: media.read_audio(s.filepath))
    check(f"{tag}: audio length ~ duration", abs(len(wave) / SR - dur) < 0.25,
          f"{len(wave) / SR:.2f}s vs {dur:.2f}s ({t:.2f}s)")
    check(f"{tag}: has_audio", media.has_audio(s.filepath))

s = vg.first()
frames, times, dur = media.read_video_frames(s.filepath, fps=1, max_frames=32, start=2, end=5)
check("window [2,5]: 3 frames at 2, 3, 4 s", len(frames) == 3 and
      np.allclose(times, [2, 3, 4], atol=1 / (s.metadata.frame_rate or 30) + 1e-3), [round(t, 3) for t in times])
wave = media.read_audio(s.filepath, start=2, end=5)
full = media.read_audio(s.filepath)
seg = full[2 * SR:5 * SR]
check("window [2,5]: audio 3.0s and matches full decode", abs(len(wave) / SR - 3) < 0.01 and
      np.corrcoef(wave[: len(seg)], seg[: len(wave)])[0, 1] > 0.99, f"{len(wave) / SR:.3f}s")

# ----------------------------------------------------------------- VidMuse + MCAP
print("\n== VidMuse-V2M + VidMuse-V2M-MCAP")
vm = fo.load_dataset("VidMuse-V2M")
mc = fo.load_dataset("VidMuse-V2M-MCAP")
ids = vm.select_group_slices("video").take(3, seed=51).values("youtube_id")
vids = dict(zip(*vm.select_group_slices("video").values(["youtube_id", "filepath"])))
mus = dict(zip(*vm.select_group_slices("music").values(["youtube_id", "filepath"])))
mcp = dict(zip(*mc.values(["youtube_id", "filepath"])))

model = foz.load_zoo_model("google/embeddinggemma-2", modalities=["vision", "audio"])
native_ok = importlib.util.find_spec("torchcodec") is not None


def native_parity(tag, path, frames):
    """media.py frames vs the processor decoding the path itself."""
    if not native_ok:
        print(f"[SKIP] {tag}: native parity (torchcodec not installed)")
        return
    tok = model._processor.video_token
    e_nat = model._forward(text=[tok], videos=[[path]])[0]
    e_ours = model.embed_frames([frames])[0]
    c = float(e_nat @ e_ours)
    check(f"{tag}: parity vs native processor path (>=0.99)", c >= 0.99, f"{c:.5f}")


for s in vg:
    fr, _, _ = media.read_video_frames(s.filepath, fps=1, max_frames=32)
    native_parity(f"VGGSound {s.youtube_id}", s.filepath, fr)

for y in ids:
    print(f"\n  {y}")
    meta = vm.select_group_slices("video").match({"youtube_id": y}).first().metadata
    (fv, tv, dv), t1 = timed(lambda: media.load_frames(vids[y], fps=1, max_frames=32))
    (fm, tm, dm), t2 = timed(lambda: media.load_frames(mcp[y], fps=1, max_frames=32))
    print(f"    mp4: {dv:.2f}s, {len(fv)} frames in {t1:.2f}s | mcap: {dm:.2f}s, {len(fm)} frames in {t2:.2f}s")
    frame_checks(f"{y} mp4", fv, tv, dv, 1, 32, meta.frame_rate)
    frame_checks(f"{y} mcap", fm, tm, dm, 1, 32, meta.frame_rate)
    native_parity(f"{y} mp4", vids[y], fv)
    check(f"{y}: has_audio(mp4) is False", not media.has_audio(vids[y]))
    check(f"{y}: mcap topics auto-detected", media.mcap_topics(mcp[y]) ==
          {"/music": "foxglove.RawAudio", "/video": "foxglove.CompressedVideo"}, media.mcap_topics(mcp[y]))

    wf, t3 = timed(lambda: media.load_audio(mus[y], max_seconds=30))
    wm, t4 = timed(lambda: media.load_audio(mcp[y], max_seconds=30))
    n = min(len(wf), len(wm))
    print(f"    wav audio {len(wf) / SR:.2f}s in {t3:.2f}s | mcap audio {len(wm) / SR:.2f}s in {t4:.2f}s")
    check(f"{y}: waveform parity wav vs mcap", np.abs(wf[:n] - wm[:n]).max() < 1e-3 and abs(len(wf) - len(wm)) <= 1,
          f"max abs diff {np.abs(wf[:n] - wm[:n]).max():.2e}, len diff {len(wf) - len(wm)}")

    ev = model.embed_frames([fv, fm])
    ea = model.embed_audio([wf, wm])
    check(f"{y}: video embedding parity mp4 vs mcap (>=0.99)", cos(ev[0], ev[1]) >= 0.99, f"{cos(ev[0], ev[1]):.4f}")
    check(f"{y}: audio embedding parity wav vs mcap (>=0.999)", cos(ea[0], ea[1]) >= 0.999, f"{cos(ea[0], ea[1]):.5f}")

print(f"\n{sum(results)}/{len(results)} passed")
