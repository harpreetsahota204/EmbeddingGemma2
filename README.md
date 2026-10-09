# EmbeddingGemma 2 for FiftyOne

A [FiftyOne](https://docs.voxel51.com/) remote zoo model for [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2), Google DeepMind's 740M-parameter embedding model that puts text, images, video and audio in one 768-dimensional space.

One model, one vector space: search images, video clips, audio files and MCAP recordings with the same text query, compare a video against its soundtrack, or classify any of them zero-shot.

## Features

- **Image and video embeddings** through `dataset.compute_embeddings()`, including clip views
- **Audio, video+audio and MCAP embeddings** through `model.embed_dataset()`, for the media FiftyOne's built-in methods don't accept
- **Text search** in Python and in the App via `fiftyone.brain` similarity indexes
- **Zero-shot classification** with `dataset.apply_model()` (image, video) or `model.classify_dataset()` (audio, MCAP)
- **Matryoshka dimensions** 768 / 512 / 256 / 128
- **Runtime settings**: frame rate, frame count, vision token budget, audio length, prompts and classes change without reloading weights
- **Selective loading**: only the encoder you need is loaded (439M parameters for image/video, 577M for audio)

## Installation

```bash
pip install fiftyone "transformers>=5.18" torch av huggingface-hub

# only for .mcap files
pip install mcap mcap-protobuf-support
```

Runs on CUDA (bfloat16) or CPU (float32). float16 is refused: the model returns NaN or degraded vectors in float16.

## Quick start

```python
import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz

foz.register_zoo_model_source(
    "https://github.com/harpreetsahota204/EmbeddingGemma2", overwrite=True
)

dataset = foz.load_zoo_dataset("quickstart")

model = foz.load_zoo_model(
    "google/embeddinggemma-2", media_type=dataset.media_type
)
dataset.compute_embeddings(model, embeddings_field="eg2")

fob.compute_similarity(
    dataset,
    embeddings="eg2",
    model="google/embeddinggemma-2",
    model_kwargs={"media_type": "image"},
    brain_key="eg2_sim",
)

view = dataset.sort_by_similarity("a photo of an airplane", k=10, brain_key="eg2_sim")
session = fo.launch_app(view)
```

## Which method to call

FiftyOne's `compute_embeddings()` and `apply_model()` accept image and video collections only. Everything else goes through two methods on the model:

| Your data | `media_type` | Embeddings | Zero-shot |
|---|---|---|---|
| Images | `"image"` | `dataset.compute_embeddings(model, ...)` | `dataset.apply_model(model, ...)` |
| Videos (one vector per video or clip) | `"video"` | `dataset.compute_embeddings(model, ...)` | `dataset.apply_model(model, ...)` |
| Audio files (wav, mp3, flac, m4a, ...) | `"audio"` | `model.embed_dataset(dataset, field)` | `model.classify_dataset(dataset, label_field)` |
| Audio track of videos | `"audio"` | `model.embed_dataset(dataset, field)` | `model.classify_dataset(...)` |
| Video + audio in one vector | `"video"` + both encoders | `model.embed_dataset(dataset, field, modality="video+audio")` | `model.classify_dataset(...)` |
| MCAP (`multimodal` datasets) | `"video"` or `"audio"` | `model.embed_dataset(dataset, field, modality=...)` | `model.classify_dataset(...)` |

### Why `media_type` is a parameter

FiftyOne chooses how to feed a model from `model.media_type`, and `load_zoo_model()` never sees your dataset. Pass `media_type=dataset.media_type` for image and video datasets.

If you load the default image model and run `compute_embeddings()` on a video dataset, FiftyOne embeds every frame into a frame-level field, and the model logs a warning. `apply_model()` gives no such warning, so load with `media_type="video"` before zero-shot classifying videos.

## Images

```python
model = foz.load_zoo_model("google/embeddinggemma-2")  # media_type="image"

dataset.compute_embeddings(model, embeddings_field="eg2", batch_size=16)

model.classes = ["airplane", "bird", "cat", "cow", "dog", "person", "car"]
dataset.apply_model(model, label_field="eg2_zeroshot", batch_size=16)
```

Images run through FiftyOne's DataLoader with worker processes (tested with spawn workers).

## Videos

```python
model = foz.load_zoo_model("google/embeddinggemma-2", media_type="video")

dataset.compute_embeddings(model, embeddings_field="eg2")

# clips are embedded over their own time window
clips = dataset.to_clips("events")
clips.compute_embeddings(model, embeddings_field="eg2")
```

Frames are decoded with PyAV and sampled by timestamp the same way the model's own processor does: one frame per second (`fps`), and if that exceeds `max_frames` (32), 32 frames spread evenly over the whole video. Long videos are therefore covered end to end.

## Audio

```python
audio = foz.load_zoo_model("google/embeddinggemma-2", media_type="audio")

audio.embed_dataset(dataset, "eg2_audio")

fob.compute_similarity(
    dataset,
    embeddings="eg2_audio",
    model="google/embeddinggemma-2",
    model_kwargs={"media_type": "audio"},
    brain_key="eg2_audio_sim",
)
view = dataset.sort_by_similarity("a dog barking", k=10, brain_key="eg2_audio_sim")

audio.classes = ["a dog barking", "rain falling", "a siren wailing"]
audio.classify_dataset(dataset, "eg2_zeroshot", embeddings_field="eg2_audio")
```

`embed_dataset()` reads `filepath`, picks the decoder by extension, writes in batches, and stores `None` (with a logged warning) for files that fail, such as a video without an audio track. FiftyOne usually reports wav-only datasets as media type `unknown`; that's fine.

Audio is resampled to 16 kHz mono. By default only the first 30 seconds are used (`max_audio_seconds=30`, the model's own feature-extractor default). For longer audio, either raise `max_audio_seconds` (up to about 320 s fits the context; quality beyond 30 s is untested) or set `long_media="chunk_mean"` to average 30 s chunks.

## Video and audio together

```python
model = foz.load_zoo_model(
    "google/embeddinggemma-2", media_type="video", modalities=["vision", "audio"]
)

model.embed_dataset(dataset, "eg2_av", modality="video+audio")
```

Frames and audio go into one input (`<|video|> <|audio|>`) and come out as one vector. Video keeps priority in the 8192-token context; audio gets the remaining room up to `max_audio_seconds`.

## MCAP

MCAP datasets have media type `multimodal`, so use `embed_dataset()`:

```python
model = foz.load_zoo_model(
    "google/embeddinggemma-2", media_type="video", modalities=["vision", "audio"]
)

model.embed_dataset(mcap_dataset, "eg2_video", modality="video")
model.embed_dataset(mcap_dataset, "eg2_audio", modality="audio")
model.embed_dataset(mcap_dataset, "eg2_av", modality="video+audio")
```

Topics are detected from their schema: the first `foxglove.CompressedVideo` or `foxglove.CompressedImage` topic for frames, the first `foxglove.RawAudio` topic (pcm-s16, pcm-s32, pcm-f32) for audio. Set `model.mcap_video_topic` / `model.mcap_audio_topic` to choose others. MCAP audio goes through the same resampler as audio files, so an MCAP recording and the wav it came from give the same vector.

## Grouped datasets

Embed one slice at a time:

```python
video_model.embed_dataset(dataset.select_group_slices("video"), "eg2", modality="video")
audio_model.embed_dataset(dataset.select_group_slices("music"), "eg2")
```

Because all vectors share one space, you can query one slice's index with another slice's vectors, for example find the music that matches a video:

```python
music = dataset.select_group_slices("music")
fob.compute_similarity(
    music, embeddings="eg2", model="google/embeddinggemma-2",
    model_kwargs={"media_type": "audio"}, brain_key="eg2_music",
)
video_vector = dataset.select_group_slices("video").first()["eg2"]
matches = music.sort_by_similarity(video_vector, k=5, brain_key="eg2_music")
```

## Text search in the App

Pass the **zoo model name** (not a model object) to `compute_similarity()`, together with `model_kwargs`. FiftyOne stores both and reloads the model when you type a query in the App. With a model object, the index works in the current Python session only.

Include `embedding_dim` in `model_kwargs` if you changed it: queries must use the same dimension as the stored vectors.

## Parameters

Set at load time with `load_zoo_model(..., name=value)`. Everything except the first four can also be changed later as a property (`model.fps = 0.5`) without reloading weights.

| Parameter | Default | Description |
|---|---|---|
| `media_type` | `"image"` | `"image"`, `"video"` or `"audio"` (load-time only) |
| `modalities` | encoder for `media_type` | encoders to load: subset of `["vision", "audio"]` (load-time only) |
| `dtype` | bfloat16 on CUDA, else float32 | `"bfloat16"` or `"float32"` (load-time only) |
| `device` | CUDA, MPS, then CPU | load-time only |
| `prompt_name` | `"SearchQuery"` | text prefix for queries: `SearchQuery`, `Classification`, `Clustering`, `SentenceSimilarity`, `QuestionAnswering`, `FactChecking`, `CodeRetrieval`, `Document`, or `None` |
| `embedding_dim` | `768` | 768, 512, 256 or 128; vectors are re-normalized after truncation |
| `image_max_soft_tokens` | `280` | tokens per image: 70, 140, 280, 560 or 1120 |
| `video_max_soft_tokens` | `140` | tokens per video frame: same choices |
| `fps` | `1` | video frames sampled per second |
| `max_frames` | `32` | frames per video, reduced automatically if they don't fit the context |
| `max_audio_seconds` | `30` | audio longer than this is truncated |
| `long_media` | `"truncate"` | `"truncate"` or `"chunk_mean"` (audio) |
| `chunk_seconds` | `30` | chunk length for `chunk_mean` |
| `mcap_video_topic`, `mcap_audio_topic` | auto | MCAP topics |
| `classes` | `None` | class names for zero-shot classification |
| `text_prompt` | `"a photo of a {}"` (image), `"{}"` (video, audio) | template for class prompts |
| `logit_scale` | `100` | softmax temperature for zero-shot confidences |

## Limits

- **Context.** Text, frames and audio share 8192 tokens. Per item: about 280 tokens per image, 140 per video frame, 25 per second of audio. At the defaults, at most 57 video frames fit; requests above that are reduced with a warning.
- **Audio-only on visually-defined classes is weak.** On 30 VGGSound test clips (309 classes), audio alone ranked the true class first 13% of the time (video: 40%, video+audio: 37%). On ESC-50, whose classes are sounds, text-to-audio precision@5 was 0.83. Use video or video+audio when labels describe what is seen.
- **128 dimensions** degrades multimodal quality a lot (MMEB 59.0 → 45.7 per the model card).
- **MCAP video** frame times assume no B-frames in the stream.
- **Untested**: cloud paths (`s3://`, `gs://`) and FiftyOne Enterprise, RRD files, ROS bags, LeRobot datasets.

## Validation

Results measured while building this integration:

| Check | Result |
|---|---|
| Parity with sentence-transformers (text, image, audio) | cos ≥ 0.99997 |
| Video frames vs the processor's own path decoding | cos 0.994–0.9997 |
| MCAP vs mp4 / wav | video cos 0.995–0.997, audio 1.0 |
| quickstart text→image precision@5 / 7-class zero-shot | 0.95 / 0.87 |
| ESC-50 text→audio precision@5 / zero-shot | 0.83 / 0.75 |
| VGGSound 30 clips, 309-class zero-shot top-1 / top-5 | video 0.40 / 0.67, audio 0.13 / 0.23, video+audio 0.37 / 0.70 |
| VGGSound 200 clips, still frame or soundtrack → its own clip in a video index (top-1) | image 0.99, audio 0.12 |

Sample sizes are small (3 to 204 items); treat these as smoke tests, not benchmarks.

## License

This integration is released under the Apache 2.0 license. The EmbeddingGemma 2 weights are released under [Apache 2.0](https://huggingface.co/google/embeddinggemma-2); use must follow the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).

## Links

- [EmbeddingGemma 2 on Hugging Face](https://huggingface.co/google/embeddinggemma-2)
- [EmbeddingGemma documentation](https://ai.google.dev/gemma/docs/embeddinggemma)
- [FiftyOne Model Zoo](https://docs.voxel51.com/user_guide/model_zoo/index.html)
- [FiftyOne Brain similarity](https://docs.voxel51.com/user_guide/brain.html#similarity)
