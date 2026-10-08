"""Media decoding for EmbeddingGemma 2: video frames, audio and MCAP.

Video frames are sampled by timestamp (never by frame index) and audio from
every source goes through the same PyAV resampler, so file and MCAP inputs
produce matching embeddings.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import io
import os

import av
import numpy as np
from PIL import Image

import fiftyone.core.storage as fos

AUDIO_SAMPLE_RATE = 16000

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".opus"}
MCAP_EXTS = {".mcap"}

MCAP_VIDEO_SCHEMAS = ("foxglove.CompressedVideo", "foxglove.CompressedImage")
MCAP_AUDIO_SCHEMAS = ("foxglove.RawAudio",)

_PCM_FORMATS = {
    "pcm-s16": ("<i2", 32768.0),
    "pcm-s32": ("<i4", 2147483648.0),
    "pcm-f32": ("<f4", 1.0),
}


def media_kind(path):
    """Returns ``"video"``, ``"audio"`` or ``"mcap"`` based on extension.

    Args:
        path: a filepath

    Returns:
        a string
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in MCAP_EXTS:
        return "mcap"

    raise ValueError(f"Unsupported file type '{ext}' for {path}")


def _open(path):
    if fos.is_local(path):
        return path

    return fos.open_file(path, "rb")


def sample_times(duration, fps, max_frames):
    """Returns target timestamps over ``[0, duration)``.

    Matches the model's own video processor: one timestamp every ``1 / fps``
    seconds starting at 0 (at least 1), then, if there are more than
    ``max_frames``, ``max_frames`` of them picked uniformly so long media is
    covered end to end.

    Args:
        duration: the duration in seconds
        fps: target frames per second
        max_frames: maximum number of frames

    Returns:
        a 1D float array
    """
    times = np.arange(max(1, int(duration * fps))) / fps
    if len(times) > max_frames:
        times = times[np.linspace(0, len(times) - 1, max_frames, dtype=int)]

    return times


def _frame_window(total, fps, max_frames, start, end, path):
    """Clamps an optional ``[start, end]`` window to ``[0, total]`` and
    returns ``(start, targets, duration)``.
    """
    start = max(0.0, start or 0.0)
    end = min(total, end) if end is not None else total
    duration = end - start
    if duration <= 0:
        raise ValueError(f"Empty window [{start}, {end}] for {path}")

    return start, start + sample_times(duration, fps, max_frames), duration


def _audio_window(start, end, max_seconds):
    """Returns ``(start, max_seconds)`` with ``max_seconds`` capped at the
    length of an optional ``[start, end]`` window.
    """
    start = max(0.0, start or 0.0)
    if end is not None:
        window = end - start
        max_seconds = window if max_seconds is None else min(window, max_seconds)

    return start, max_seconds


def _pick_at(timed_items, targets, convert):
    """Picks, for each target time, the item showing at that time (the last
    item with ``time <= target``) from a time-ordered iterable of
    ``(time, item)``. Always returns ``len(targets)`` results; targets
    before the first item use the first item and targets past the end reuse
    the last one.
    """
    targets = list(targets)
    picked = [None] * len(targets)
    times = [None] * len(targets)

    # holds a reference to the item so its id() cannot be reused
    cache = [None, None]

    def _get(t, item):
        if cache[0] is not item:
            cache[0] = item
            cache[1] = convert(item)
        return cache[1]

    eps = 1e-6
    j = 0
    prev = None
    for t, item in timed_items:
        while j < len(targets) and t > targets[j] + eps:
            chosen = prev if prev is not None else (t, item)
            picked[j], times[j] = _get(*chosen), chosen[0]
            j += 1

        if j == len(targets):
            break

        prev = (t, item)

    if prev is None and j == 0:
        raise ValueError("No frames were decoded")

    while j < len(targets):
        picked[j], times[j] = _get(*prev), prev[0]
        j += 1

    return picked, times


###############################################################################
# Video files
###############################################################################


def _video_duration(container, stream):
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)

    if container.duration is not None:
        return container.duration / av.time_base

    raise ValueError("Could not determine the video duration")


def read_video_frames(path, fps=1, max_frames=32, start=None, end=None):
    """Samples frames from a video file by timestamp.

    Args:
        path: a video filepath
        fps (1): target frames per second
        max_frames (32): maximum number of frames
        start (None): optional window start in seconds
        end (None): optional window end in seconds

    Returns:
        a tuple of

        -   frames: a list of PIL images
        -   times: the timestamp (seconds, relative to the stream start) of
            each frame
        -   duration: the duration of the sampled window in seconds
    """
    with av.open(_open(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        total = _video_duration(container, stream)
        t0 = float(stream.start_time * stream.time_base) if stream.start_time else 0.0
        start, targets, duration = _frame_window(
            total, fps, max_frames, start, end, path
        )

        if start > 0:
            container.seek(
                int((start + t0) / stream.time_base),
                stream=stream,
                backward=True,
            )

        def _timed():
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                yield float(frame.pts * stream.time_base) - t0, frame

        frames, times = _pick_at(
            _timed(), targets, lambda f: f.to_image()
        )

    return frames, times, duration


###############################################################################
# Audio files
###############################################################################


def _resample(frames, max_seconds=None):
    """Resamples an iterable of ``av.AudioFrame`` to 16 kHz mono float32."""
    resampler = av.AudioResampler(
        format="flt", layout="mono", rate=AUDIO_SAMPLE_RATE
    )
    max_samples = (
        int(max_seconds * AUDIO_SAMPLE_RATE) if max_seconds is not None else None
    )

    chunks = []
    n = 0
    for frame in frames:
        for out in resampler.resample(frame):
            chunk = out.to_ndarray().reshape(-1)
            chunks.append(chunk)
            n += len(chunk)

        if max_samples is not None and n >= max_samples:
            break
    else:
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))

    if not chunks:
        return np.zeros(0, dtype=np.float32)

    wave = np.concatenate(chunks).astype(np.float32)
    if max_samples is not None:
        wave = wave[:max_samples]

    return wave


def has_audio(path):
    """Whether a media file (or MCAP) contains audio.

    Args:
        path: a filepath

    Returns:
        True/False
    """
    if media_kind(path) == "mcap":
        return bool(mcap_topics(path, MCAP_AUDIO_SCHEMAS))

    with av.open(_open(path)) as container:
        return len(container.streams.audio) > 0


def read_audio(path, max_seconds=None, start=None, end=None):
    """Decodes the first audio stream of a file to 16 kHz mono float32.

    Works for audio files and for the audio track of video files.

    Args:
        path: a filepath
        max_seconds (None): stop after this many seconds
        start (None): optional window start in seconds
        end (None): optional window end in seconds

    Returns:
        a 1D float32 array
    """
    with av.open(_open(path)) as container:
        if not container.streams.audio:
            raise ValueError(f"No audio stream in {path}")

        stream = container.streams.audio[0]
        start, max_seconds = _audio_window(start, end, max_seconds)

        if start > 0:
            container.seek(int(start * av.time_base), backward=True)

        first_time = []

        def _frames():
            for frame in container.decode(stream):
                if not first_time:
                    first_time.append(frame.time if frame.time is not None else 0.0)
                yield frame

        # decode a little extra so the seek offset can be trimmed
        extra = 10.0 if start > 0 else 0.0
        wave = _resample(
            _frames(), None if max_seconds is None else max_seconds + extra
        )

    if start > 0 and first_time:
        offset = int(round((start - first_time[0]) * AUDIO_SAMPLE_RATE))
        wave = wave[max(0, offset) :]

    if max_seconds is not None:
        wave = wave[: int(max_seconds * AUDIO_SAMPLE_RATE)]

    return wave


###############################################################################
# MCAP
###############################################################################


def _mcap_reader(f):
    try:
        from mcap.reader import make_reader
        from mcap_protobuf.decoder import DecoderFactory
    except ImportError as e:
        raise ImportError(
            "Reading .mcap files requires `pip install mcap "
            "mcap-protobuf-support`"
        ) from e

    return make_reader(f, decoder_factories=[DecoderFactory()])


def mcap_topics(path, schemas=None):
    """Returns ``{topic: schema_name}`` for an MCAP file.

    Args:
        path: an MCAP filepath
        schemas (None): optional schema names to keep

    Returns:
        a dict, ordered by channel ID
    """
    with fos.open_file(path, "rb") as f:
        summary = _mcap_reader(f).get_summary()

    topics = {}
    for cid in sorted(summary.channels):
        channel = summary.channels[cid]
        schema = summary.schemas.get(channel.schema_id)
        name = schema.name if schema is not None else None
        if schemas is None or name in schemas:
            topics[channel.topic] = name

    return topics


def _resolve_topic(path, topic, schemas, kind):
    topics = mcap_topics(path)
    if topic is not None:
        if topic not in topics:
            raise ValueError(
                f"Topic '{topic}' not found in {path}; available: "
                f"{sorted(topics)}"
            )
        if topics[topic] not in schemas:
            raise ValueError(
                f"Topic '{topic}' has schema '{topics[topic]}'; {kind} "
                f"requires one of {schemas}"
            )
        return topic, topics[topic]

    for t, schema in topics.items():
        if schema in schemas:
            return t, schema

    raise ValueError(
        f"No {kind} topic in {path}. Expected a schema in {schemas}; found "
        f"{topics}"
    )


def read_mcap_frames(
    path, fps=1, max_frames=32, topic=None, start=None, end=None
):
    """Samples frames from an MCAP video or image topic by timestamp.

    Supports ``foxglove.CompressedVideo`` (decoded with PyAV) and
    ``foxglove.CompressedImage``. Times are message log times relative to
    the first message on the topic.

    Args:
        path: an MCAP filepath
        fps (1): target frames per second
        max_frames (32): maximum number of frames
        topic (None): the topic to read. Defaults to the first video or image
            topic
        start (None): optional window start in seconds
        end (None): optional window end in seconds

    Returns:
        a tuple of ``(frames, times, duration)`` as in
        :func:`read_video_frames`
    """
    topic, schema = _resolve_topic(path, topic, MCAP_VIDEO_SCHEMAS, "video")

    with fos.open_file(path, "rb") as f:
        reader = _mcap_reader(f)
        log_times = [
            m.log_time
            for _, _, m in reader.iter_messages(
                topics=[topic], log_time_order=True
            )
        ]

    if not log_times:
        raise ValueError(f"Topic '{topic}' in {path} has no messages")

    t0 = log_times[0]
    rel = np.array([(t - t0) / 1e9 for t in log_times])
    period = float(np.median(np.diff(rel))) if len(rel) > 1 else 0.0
    total = rel[-1] + period

    _, targets, duration = _frame_window(
        total, fps, max_frames, start, end, path
    )

    with fos.open_file(path, "rb") as f:
        reader = _mcap_reader(f)
        messages = reader.iter_decoded_messages(
            topics=[topic], log_time_order=True
        )

        if schema == "foxglove.CompressedImage":
            timed = (((m.log_time - t0) / 1e9, d) for _, _, m, d in messages)
            frames, times = _pick_at(
                timed,
                targets,
                lambda d: Image.open(io.BytesIO(d.data)).convert("RGB"),
            )
        else:
            frames, times = _pick_at(
                _decode_compressed_video(messages, t0),
                targets,
                lambda fr: fr.to_image(),
            )

    return frames, times, duration


def _decode_compressed_video(messages, t0):
    codec = None
    pending = []
    for _, _, message, decoded in messages:
        if codec is None:
            codec = av.CodecContext.create(decoded.format, "r")

        pending.append((message.log_time - t0) / 1e9)
        for frame in codec.decode(av.Packet(decoded.data)):
            yield pending.pop(0), frame

    if codec is not None:
        for frame in codec.decode(None):
            if pending:
                yield pending.pop(0), frame


def read_mcap_audio(path, max_seconds=None, topic=None, start=None, end=None):
    """Decodes an MCAP ``foxglove.RawAudio`` topic to 16 kHz mono float32.

    PCM is wrapped in ``av.AudioFrame`` objects and resampled with the same
    PyAV path as :func:`read_audio`.

    Args:
        path: an MCAP filepath
        max_seconds (None): stop after this many seconds
        topic (None): the topic to read. Defaults to the first audio topic
        start (None): optional window start in seconds
        end (None): optional window end in seconds

    Returns:
        a 1D float32 array
    """
    topic, _ = _resolve_topic(path, topic, MCAP_AUDIO_SCHEMAS, "audio")
    start, max_seconds = _audio_window(start, end, max_seconds)

    def _frames(reader):
        t0 = None
        for _, _, message, d in reader.iter_decoded_messages(
            topics=[topic], log_time_order=True
        ):
            if t0 is None:
                t0 = message.log_time

            if d.format not in _PCM_FORMATS:
                raise ValueError(
                    f"Unsupported RawAudio format '{d.format}'; supported: "
                    f"{sorted(_PCM_FORMATS)}"
                )

            dtype, scale = _PCM_FORMATS[d.format]
            channels = d.number_of_channels
            pcm = np.frombuffer(d.data, dtype=dtype).astype(np.float32) / scale
            pcm = pcm.reshape(-1, channels)

            msg_start = (message.log_time - t0) / 1e9
            msg_end = msg_start + len(pcm) / d.sample_rate
            if msg_end <= start:
                continue
            if msg_start < start:
                pcm = pcm[int(round((start - msg_start) * d.sample_rate)) :]

            frame = av.AudioFrame.from_ndarray(
                np.ascontiguousarray(pcm.T),
                format="fltp",
                layout=_layout(channels),
            )
            frame.sample_rate = d.sample_rate
            yield frame

    with fos.open_file(path, "rb") as f:
        return _resample(_frames(_mcap_reader(f)), max_seconds)


def _layout(channels):
    if channels == 1:
        return "mono"
    if channels == 2:
        return "stereo"

    raise ValueError(f"Unsupported channel count: {channels}")


###############################################################################
# Dispatch by file type
###############################################################################


def load_frames(path, fps=1, max_frames=32, topic=None, start=None, end=None):
    """Samples frames from a video file or MCAP by timestamp.

    Args:
        path: a filepath
        fps (1): target frames per second
        max_frames (32): maximum number of frames
        topic (None): MCAP topic (ignored for video files)
        start (None): optional window start in seconds
        end (None): optional window end in seconds

    Returns:
        a tuple of ``(frames, times, duration)``
    """
    kind = media_kind(path)
    if kind == "video":
        return read_video_frames(
            path, fps=fps, max_frames=max_frames, start=start, end=end
        )
    if kind == "mcap":
        return read_mcap_frames(
            path,
            fps=fps,
            max_frames=max_frames,
            topic=topic,
            start=start,
            end=end,
        )

    raise ValueError(f"{path} has no video")


def load_audio(path, max_seconds=None, topic=None, start=None, end=None):
    """Decodes audio from an audio file, video file or MCAP.

    Args:
        path: a filepath
        max_seconds (None): stop after this many seconds
        topic (None): MCAP topic (ignored for other files)
        start (None): optional window start in seconds
        end (None): optional window end in seconds

    Returns:
        a 1D float32 array, 16 kHz mono
    """
    if media_kind(path) == "mcap":
        return read_mcap_audio(
            path, max_seconds=max_seconds, topic=topic, start=start, end=end
        )

    return read_audio(path, max_seconds=max_seconds, start=start, end=end)
