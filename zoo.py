"""EmbeddingGemma 2 models for FiftyOne.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

# Worker-pickle constraint: nothing in this module may be pickled across
# DataLoader workers. Use ImageGetItem and the inherited collate_fn only.

import logging
import math

import eta.core.frameutils as etaf
import eta.core.video as etav
import numpy as np
from PIL import Image
import torch
from transformers import AutoConfig, AutoModel, AutoProcessor

import fiftyone.core.fields as fof
import fiftyone.core.labels as fol
import fiftyone.core.models as fom
import fiftyone.core.utils as fou
import fiftyone.utils.torch as fout
from fiftyone.utils.torch import ImageGetItem

from . import media

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "google/embeddinggemma-2"

# config_sentence_transformers.json; prefixes apply to text only
PROMPTS = {
    "BitextMining": "task: search result | query: ",
    "Classification": "task: classification | query: ",
    "Clustering": "task: clustering | query: ",
    "CodeRetrieval": "task: code retrieval | query: ",
    "Document": "title: none | text: ",
    "FactChecking": "task: fact checking | query: ",
    "InstructionRetrieval": "task: code retrieval | query: ",
    "MultilabelClassification": "task: classification | query: ",
    "PairClassification": "task: sentence similarity | query: ",
    "QuestionAnswering": "task: question answering | query: ",
    "Reranking": "task: search result | query: ",
    "Retrieval": "task: search result | query: ",
    "Retrieval-document": "title: none | text: ",
    "Retrieval-query": "task: search result | query: ",
    "STS": "task: sentence similarity | query: ",
    "SearchQuery": "task: search result | query: ",
    "SentenceSimilarity": "task: sentence similarity | query: ",
    "Summarization": "task: sentence similarity | query: ",
    "document": "title: none | text: ",
    "query": "task: search result | query: ",
}

VALID_MEDIA_TYPES = ("image", "video", "audio")
VALID_DATASET_MODALITIES = ("video", "audio", "video+audio")
VALID_MODALITIES = ("vision", "audio")
VALID_EMBEDDING_DIMS = (128, 256, 512, 768)
VALID_SOFT_TOKENS = (70, 140, 280, 560, 1120)
VALID_LONG_MEDIA = ("truncate", "chunk_mean")
VALID_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}

MAX_CONTEXT_TOKENS = 8192
AUDIO_SAMPLE_RATE = 16000
AUDIO_TOKENS_PER_SECOND = 25
TEXT_BATCH_SIZE = 64

# media_type -> encoders to load (text is always loaded)
DEFAULT_MODALITIES = {
    "image": ("vision",),
    "video": ("vision",),
    "audio": ("audio",),
}

# media_type -> zero-shot class prompt template
DEFAULT_TEXT_PROMPTS = {
    "image": "a photo of a {}",
    "video": "{}",
    "audio": "{}",
}


def _validate_choice(name, value, valid):
    if value not in valid:
        raise ValueError(f"{name} must be one of {tuple(valid)}, got {value!r}")


def _validate_positive(name, value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _validate_text_prompt(value):
    if not isinstance(value, str) or "{}" not in value:
        raise ValueError(
            f"text_prompt must be a string containing '{{}}', got {value!r}"
        )


def _validate_classes(value):
    if value is None:
        return
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(c, str) for c in value
    ):
        raise TypeError("classes must be a list of strings or None")


def _to_pil(img):
    if isinstance(img, Image.Image):
        return img.convert("RGB")

    if isinstance(img, str):
        return Image.open(img).convert("RGB")

    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))

    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = np.clip(img * 255 if img.max() <= 1.0 else img, 0, 255)
        img = img.astype(np.uint8)

    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]

    return Image.fromarray(img).convert("RGB")


class EmbeddingGemma2Config(fout.TorchImageModelConfig):
    """Configuration for EmbeddingGemma 2 models.

    Construction-only (changing these reloads weights): ``model_path``,
    ``media_type``, ``modalities``, ``dtype``, ``device``.

    Everything else is a runtime parameter with a setter on the model.

    Args:
        model_path ("google/embeddinggemma-2"): a local directory or Hugging
            Face repo ID
        media_type ("image"): ``"image"``, ``"video"`` or ``"audio"``
        modalities (None): encoders to load in addition to text, a subset of
            ``("vision", "audio")``. Defaults to the encoder ``media_type``
            needs
        dtype (None): ``"bfloat16"`` or ``"float32"``. Defaults to bfloat16
            on CUDA devices that support it, else float32. float16 is not
            supported
        prompt_name ("SearchQuery"): text prefix for queries, a key of
            :data:`PROMPTS`, or None for no prefix
        embedding_dim (768): Matryoshka dimension, one of 128/256/512/768
        image_max_soft_tokens (280): vision tokens per image
        video_max_soft_tokens (140): vision tokens per video frame
        fps (1): video frames sampled per second
        max_frames (32): maximum video frames per sample
        max_audio_seconds (30): audio longer than this is truncated (or
            chunked, see ``long_media``)
        long_media ("truncate"): ``"truncate"`` or ``"chunk_mean"``
        chunk_seconds (30): chunk length for ``long_media="chunk_mean"``
        mcap_video_topic (None): MCAP topic for frames. Defaults to the
            first ``foxglove.CompressedVideo``/``CompressedImage`` topic
        mcap_audio_topic (None): MCAP topic for audio. Defaults to the first
            ``foxglove.RawAudio`` topic
        classes (None): class names for zero-shot classification
        text_prompt (None): template for class prompts. Defaults to
            ``"a photo of a {}"`` for images and ``"{}"`` for video and audio
        logit_scale (100): softmax temperature for zero-shot scores
    """

    def __init__(self, d):
        if "raw_inputs" not in d:
            d["raw_inputs"] = True

        super().__init__(d)

        self.model_path = self.parse_string(
            d, "model_path", default=DEFAULT_MODEL_PATH
        )

        self.media_type = self.parse_string(d, "media_type", default="image")
        _validate_choice("media_type", self.media_type, VALID_MEDIA_TYPES)

        self.modalities = self.parse_array(d, "modalities", default=None)
        if self.modalities is None:
            self.modalities = list(DEFAULT_MODALITIES[self.media_type])
        for m in self.modalities:
            _validate_choice("modalities", m, VALID_MODALITIES)

        self.dtype = self.parse_string(d, "dtype", default=None)
        if self.dtype in ("float16", "fp16", "half") or self.use_half_precision:
            raise ValueError(
                "EmbeddingGemma 2 does not support float16 (it returns NaN or "
                "degraded vectors). Use dtype='bfloat16' or 'float32'"
            )
        if self.dtype is not None:
            _validate_choice("dtype", self.dtype, VALID_DTYPES)

        self.prompt_name = self.parse_string(
            d, "prompt_name", default="SearchQuery"
        )
        if self.prompt_name is not None:
            _validate_choice("prompt_name", self.prompt_name, PROMPTS)

        self.embedding_dim = self.parse_int(d, "embedding_dim", default=768)
        _validate_choice(
            "embedding_dim", self.embedding_dim, VALID_EMBEDDING_DIMS
        )

        self.image_max_soft_tokens = self.parse_int(
            d, "image_max_soft_tokens", default=280
        )
        _validate_choice(
            "image_max_soft_tokens",
            self.image_max_soft_tokens,
            VALID_SOFT_TOKENS,
        )

        self.video_max_soft_tokens = self.parse_int(
            d, "video_max_soft_tokens", default=140
        )
        _validate_choice(
            "video_max_soft_tokens",
            self.video_max_soft_tokens,
            VALID_SOFT_TOKENS,
        )

        self.fps = self.parse_number(d, "fps", default=1)
        _validate_positive("fps", self.fps)

        self.max_frames = self.parse_int(d, "max_frames", default=32)
        _validate_positive("max_frames", self.max_frames)

        self.max_audio_seconds = self.parse_number(
            d, "max_audio_seconds", default=30
        )
        _validate_positive("max_audio_seconds", self.max_audio_seconds)

        self.long_media = self.parse_string(d, "long_media", default="truncate")
        _validate_choice("long_media", self.long_media, VALID_LONG_MEDIA)

        self.chunk_seconds = self.parse_number(d, "chunk_seconds", default=30)
        _validate_positive("chunk_seconds", self.chunk_seconds)

        self.mcap_video_topic = self.parse_string(
            d, "mcap_video_topic", default=None
        )
        self.mcap_audio_topic = self.parse_string(
            d, "mcap_audio_topic", default=None
        )

        _validate_classes(self.classes)

        self.text_prompt = self.parse_string(
            d, "text_prompt", default=DEFAULT_TEXT_PROMPTS[self.media_type]
        )
        _validate_text_prompt(self.text_prompt)

        self.logit_scale = self.parse_number(d, "logit_scale", default=100)
        _validate_positive("logit_scale", self.logit_scale)


class EmbeddingGemma2Base(object):
    """Media-agnostic EmbeddingGemma 2 core.

    Owns the weights, processor, device and dtype, and the
    :class:`fiftyone.core.models.Model` members shared by every media type.
    All ``embed_*`` methods return float32, L2-normalized arrays of shape
    ``(num_inputs, embedding_dim)``.

    Args:
        config: an :class:`EmbeddingGemma2Config`
    """

    def __init__(self, config):
        self.config = config
        self._device = self._resolve_device(config.device)
        self._dtype = self._resolve_dtype(config.dtype, self._device)
        self._processor = AutoProcessor.from_pretrained(config.model_path)
        self._model = self._load_model()
        self._class_embeddings = None
        self._class_embeddings_key = None
        self._preprocess = False

    @staticmethod
    def _resolve_device(device):
        if device is not None:
            return torch.device(device)

        if torch.cuda.is_available():
            return torch.device("cuda")

        if torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    @staticmethod
    def _resolve_dtype(dtype, device):
        if dtype is not None:
            return VALID_DTYPES[dtype]

        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16

        return torch.float32

    def _load_model(self):
        hf_config = AutoConfig.from_pretrained(self.config.model_path)
        if "vision" not in self.config.modalities:
            hf_config.vision_config = None
        if "audio" not in self.config.modalities:
            hf_config.audio_config = None

        model = AutoModel.from_pretrained(
            self.config.model_path, config=hf_config, dtype=self._dtype
        )
        return model.to(self._device).eval()

    @property
    def device(self):
        """The ``torch.device`` the model runs on."""
        return self._device

    @property
    def dtype(self):
        """The ``torch.dtype`` of the weights."""
        return self._dtype

    @property
    def has_embeddings(self):
        return True

    @property
    def can_embed_prompts(self):
        return True

    @property
    def transforms(self):
        return None

    @property
    def preprocess(self):
        return self._preprocess

    @preprocess.setter
    def preprocess(self, value):
        self._preprocess = value

    @property
    def ragged_batches(self):
        return False

    def embed(self, arg):
        return self.embed_all([arg])[0]

    def predict(self, arg):
        return self.predict_all([arg])[0]

    ###########################################################################
    # Runtime parameters
    ###########################################################################

    @property
    def prompt_name(self):
        """Text prefix used for queries; a key of :data:`PROMPTS` or None."""
        return self.config.prompt_name

    @prompt_name.setter
    def prompt_name(self, value):
        if value is not None:
            _validate_choice("prompt_name", value, PROMPTS)
        self.config.prompt_name = value

    @property
    def embedding_dim(self):
        """Matryoshka output dimension (128, 256, 512 or 768)."""
        return self.config.embedding_dim

    @embedding_dim.setter
    def embedding_dim(self, value):
        _validate_choice("embedding_dim", value, VALID_EMBEDDING_DIMS)
        self.config.embedding_dim = value

    @property
    def image_max_soft_tokens(self):
        """Vision tokens per image (70, 140, 280, 560 or 1120)."""
        return self.config.image_max_soft_tokens

    @image_max_soft_tokens.setter
    def image_max_soft_tokens(self, value):
        _validate_choice("image_max_soft_tokens", value, VALID_SOFT_TOKENS)
        self.config.image_max_soft_tokens = value

    @property
    def video_max_soft_tokens(self):
        """Vision tokens per video frame (70, 140, 280, 560 or 1120)."""
        return self.config.video_max_soft_tokens

    @video_max_soft_tokens.setter
    def video_max_soft_tokens(self, value):
        _validate_choice("video_max_soft_tokens", value, VALID_SOFT_TOKENS)
        self.config.video_max_soft_tokens = value

    @property
    def fps(self):
        """Video frames sampled per second."""
        return self.config.fps

    @fps.setter
    def fps(self, value):
        _validate_positive("fps", value)
        self.config.fps = value

    @property
    def max_frames(self):
        """Maximum video frames per sample."""
        return self.config.max_frames

    @max_frames.setter
    def max_frames(self, value):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("max_frames must be an int")
        _validate_positive("max_frames", value)
        self.config.max_frames = value

    @property
    def max_audio_seconds(self):
        """Audio longer than this is truncated or chunked."""
        return self.config.max_audio_seconds

    @max_audio_seconds.setter
    def max_audio_seconds(self, value):
        _validate_positive("max_audio_seconds", value)
        self.config.max_audio_seconds = value

    @property
    def long_media(self):
        """``"truncate"`` or ``"chunk_mean"``."""
        return self.config.long_media

    @long_media.setter
    def long_media(self, value):
        _validate_choice("long_media", value, VALID_LONG_MEDIA)
        self.config.long_media = value

    @property
    def chunk_seconds(self):
        """Chunk length used when ``long_media="chunk_mean"``."""
        return self.config.chunk_seconds

    @chunk_seconds.setter
    def chunk_seconds(self, value):
        _validate_positive("chunk_seconds", value)
        self.config.chunk_seconds = value

    @property
    def mcap_video_topic(self):
        """MCAP topic for frames, or None to auto-detect."""
        return self.config.mcap_video_topic

    @mcap_video_topic.setter
    def mcap_video_topic(self, value):
        if value is not None and not isinstance(value, str):
            raise TypeError("mcap_video_topic must be a string or None")
        self.config.mcap_video_topic = value

    @property
    def mcap_audio_topic(self):
        """MCAP topic for audio, or None to auto-detect."""
        return self.config.mcap_audio_topic

    @mcap_audio_topic.setter
    def mcap_audio_topic(self, value):
        if value is not None and not isinstance(value, str):
            raise TypeError("mcap_audio_topic must be a string or None")
        self.config.mcap_audio_topic = value

    @property
    def classes(self):
        """Class names for zero-shot classification."""
        return self.config.classes

    @classes.setter
    def classes(self, value):
        _validate_classes(value)
        self.config.classes = list(value) if value is not None else None

    @property
    def text_prompt(self):
        """Template for class prompts; must contain ``{}``."""
        return self.config.text_prompt

    @text_prompt.setter
    def text_prompt(self, value):
        _validate_text_prompt(value)
        self.config.text_prompt = value

    @property
    def logit_scale(self):
        """Softmax temperature applied to cosine similarities."""
        return self.config.logit_scale

    @logit_scale.setter
    def logit_scale(self, value):
        _validate_positive("logit_scale", value)
        self.config.logit_scale = value

    ###########################################################################
    # Core embedding
    ###########################################################################

    def _require(self, modality):
        if modality not in self.config.modalities:
            raise ValueError(
                f"This model was loaded without the {modality} encoder. "
                f"Reload with modalities including '{modality}'"
            )

    def _audio_kwargs(self):
        max_samples = int(self.config.max_audio_seconds * AUDIO_SAMPLE_RATE)
        return {"max_length": max_samples, "truncation": True}

    def _forward(self, **processor_kwargs):
        enc = self._processor(
            return_tensors="pt", padding=True, **processor_kwargs
        )

        num_tokens = int(enc["attention_mask"].sum(dim=1).max())
        if num_tokens > MAX_CONTEXT_TOKENS:
            raise ValueError(
                f"Input has {num_tokens} tokens; the model supports "
                f"{MAX_CONTEXT_TOKENS}. Reduce frames, soft tokens or audio "
                "length"
            )

        enc = enc.to(self._device)
        with torch.inference_mode():
            out = self._model(**enc)

        return self._pool(out.last_hidden_state, enc["attention_mask"])

    def _pool(self, hidden, attention_mask):
        hidden = hidden.float()
        mask = attention_mask.unsqueeze(-1).float()
        emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        emb = emb[:, : self.config.embedding_dim]
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.cpu().numpy()

    def _prefix(self, prompt_name):
        return PROMPTS[prompt_name] if prompt_name is not None else ""

    def embed_text(self, texts, prompt_name=None):
        """Embeds text.

        Args:
            texts: a string or list of strings
            prompt_name (None): a key of :data:`PROMPTS`, or None for no
                prefix

        Returns:
            a ``(num_texts, embedding_dim)`` array
        """
        if isinstance(texts, str):
            texts = [texts]

        if prompt_name is not None:
            _validate_choice("prompt_name", prompt_name, PROMPTS)

        prefix = self._prefix(prompt_name)
        texts = [prefix + t for t in texts]

        batches = []
        for i in range(0, len(texts), TEXT_BATCH_SIZE):
            batches.append(
                self._forward(
                    text=texts[i : i + TEXT_BATCH_SIZE],
                    truncation=True,
                    max_length=MAX_CONTEXT_TOKENS,
                )
            )

        return np.concatenate(batches, axis=0)

    def embed_images(self, images):
        """Embeds images.

        Args:
            images: a list of PIL images, numpy HWC arrays, CHW tensors or
                filepaths

        Returns:
            a ``(num_images, embedding_dim)`` array
        """
        self._require("vision")
        images = [_to_pil(img) for img in images]
        token = self._processor.image_token
        return self._forward(
            text=[token] * len(images),
            images=[[img] for img in images],
            images_kwargs={"max_soft_tokens": self.config.image_max_soft_tokens},
        )

    def embed_frames(self, videos):
        """Embeds videos given as already-sampled frames.

        Args:
            videos: a list of videos, each a list of PIL images or numpy HWC
                arrays

        Returns:
            a ``(num_videos, embedding_dim)`` array
        """
        self._require("vision")
        videos = [[_to_pil(f) for f in frames] for frames in videos]
        for frames in videos:
            self._check_budget(len(frames))

        token = self._processor.video_token
        return self._forward(
            text=[token] * len(videos),
            videos=[[frames] for frames in videos],
            videos_kwargs={
                "do_sample_frames": False,
                "max_soft_tokens": self.config.video_max_soft_tokens,
            },
        )

    def embed_audio(self, waves):
        """Embeds audio.

        Audio longer than ``max_audio_seconds`` is truncated.

        Args:
            waves: a list of 1D float32 arrays, mono, 16 kHz

        Returns:
            a ``(num_waves, embedding_dim)`` array
        """
        self._require("audio")
        waves = [np.asarray(w, dtype=np.float32) for w in waves]

        max_samples = int(self.config.max_audio_seconds * AUDIO_SAMPLE_RATE)
        num_truncated = sum(len(w) > max_samples for w in waves)
        if num_truncated:
            logger.warning(
                "Truncated %d audio input(s) to max_audio_seconds=%s",
                num_truncated,
                self.config.max_audio_seconds,
            )

        token = self._processor.audio_token
        return self._forward(
            text=[token] * len(waves),
            audio=[[w] for w in waves],
            audio_kwargs=self._audio_kwargs(),
        )

    def embed_interleaved(self, items, prompt_name=None):
        """Embeds interleaved text and media, one vector per item.

        Each item is a dict with a ``"text"`` string containing
        ``<|image|>``, ``<|video|>`` and ``<|audio|>`` placeholders, plus
        optional ``"images"`` (list of images), ``"videos"`` (list of frame
        lists) and ``"audio"`` (list of 16 kHz mono waveforms) filled into
        the placeholders in order. Items are embedded one at a time.

        Args:
            items: a list of dicts
            prompt_name (None): a text prefix applied to ``"text"``

        Returns:
            a ``(num_items, embedding_dim)`` array
        """
        prefix = self._prefix(prompt_name)
        results = []
        for item in items:
            kwargs = {"text": [prefix + item.get("text", "")]}

            images = item.get("images")
            if images:
                self._require("vision")
                kwargs["images"] = [[_to_pil(img) for img in images]]
                kwargs["images_kwargs"] = {
                    "max_soft_tokens": self.config.image_max_soft_tokens
                }

            videos = item.get("videos")
            if videos:
                self._require("vision")
                kwargs["videos"] = [
                    [[_to_pil(f) for f in frames] for frames in videos]
                ]
                kwargs["videos_kwargs"] = {
                    "do_sample_frames": False,
                    "max_soft_tokens": self.config.video_max_soft_tokens,
                }

            audio = item.get("audio")
            if audio:
                self._require("audio")
                kwargs["audio"] = [
                    [np.asarray(w, dtype=np.float32) for w in audio]
                ]
                kwargs["audio_kwargs"] = self._audio_kwargs()

            results.append(self._forward(**kwargs))

        return np.concatenate(results, axis=0)

    def load_frames(self, path, start=None, end=None):
        """Samples frames from a video file or MCAP per the current ``fps``,
        ``max_frames`` and token budget.

        Args:
            path: a video or ``.mcap`` filepath
            start (None): optional window start in seconds
            end (None): optional window end in seconds

        Returns:
            a list of PIL images
        """
        max_frames, _ = self.fit_budget(num_frames=self.config.max_frames)
        frames, _, _ = media.load_frames(
            path,
            fps=self.config.fps,
            max_frames=max_frames,
            topic=self.config.mcap_video_topic,
            start=start,
            end=end,
        )
        return frames

    def _fit_frames(self, frames):
        max_frames, _ = self.fit_budget(num_frames=len(frames))
        if len(frames) <= max_frames:
            return list(frames)

        idx = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        return [frames[i] for i in idx]

    def embed_audio_file(self, path, start=None, end=None):
        """Embeds the audio of an audio file, video file or MCAP.

        With ``long_media="truncate"`` the first ``max_audio_seconds`` are
        used. With ``long_media="chunk_mean"`` the whole track is split into
        ``chunk_seconds`` chunks whose embeddings are averaged.

        Args:
            path: a filepath
            start (None): optional window start in seconds
            end (None): optional window end in seconds

        Returns:
            a ``(embedding_dim,)`` array
        """
        topic = self.config.mcap_audio_topic
        if self.config.long_media == "truncate":
            wave = media.load_audio(
                path,
                max_seconds=self.config.max_audio_seconds,
                topic=topic,
                start=start,
                end=end,
            )
            return self.embed_audio([wave])[0]

        wave = media.load_audio(path, topic=topic, start=start, end=end)
        n = int(self.config.chunk_seconds * AUDIO_SAMPLE_RATE)
        chunks = [wave[i : i + n] for i in range(0, len(wave), n)]
        if len(chunks) > 1 and len(chunks[-1]) < AUDIO_SAMPLE_RATE:
            chunks = chunks[:-1]

        vectors = np.concatenate(
            [self.embed_audio(chunks[i : i + 8]) for i in range(0, len(chunks), 8)]
        )
        mean = vectors.mean(axis=0)
        return (mean / np.linalg.norm(mean)).astype(np.float32)

    def embed_video_audio(self, path, start=None, end=None):
        """Embeds the frames and audio of a video file or MCAP as one
        interleaved input (``<|video|> <|audio|>``).

        Requires both encoders (``modalities=["vision", "audio"]``). Audio
        gets whatever context the frames leave, up to ``max_audio_seconds``.

        Args:
            path: a filepath
            start (None): optional window start in seconds
            end (None): optional window end in seconds

        Returns:
            a ``(embedding_dim,)`` array
        """
        self._require("vision")
        self._require("audio")
        frames = self.load_frames(path, start=start, end=end)
        _, audio_seconds = self.fit_budget(
            num_frames=len(frames),
            audio_seconds=self.config.max_audio_seconds,
            text_tokens=1,
        )
        wave = media.load_audio(
            path,
            max_seconds=audio_seconds,
            topic=self.config.mcap_audio_topic,
            start=start,
            end=end,
        )
        p = self._processor
        item = {
            "text": f"{p.video_token} {p.audio_token}",
            "videos": [frames],
            "audio": [wave],
        }
        return self.embed_interleaved([item])[0]

    def _embed_path(self, path, modality):
        if modality == "video":
            return self.embed_frames([self.load_frames(path)])[0]
        if modality == "audio":
            return self.embed_audio_file(path)

        return self.embed_video_audio(path)

    def _default_modality(self, modality):
        if modality is None:
            modality = {"video": "video", "audio": "audio"}.get(
                self.media_type
            )
            if modality is None:
                raise ValueError(
                    "Pass modality='video', 'audio' or 'video+audio'"
                )

        _validate_choice("modality", modality, VALID_DATASET_MODALITIES)
        if "video" in modality:
            self._require("vision")
        if "audio" in modality:
            self._require("audio")

        return modality

    def embed_dataset(
        self,
        samples,
        embeddings_field=None,
        modality=None,
        filepath_field="filepath",
        skip_existing=False,
        skip_failures=True,
        batch_size=64,
        progress=None,
    ):
        """Embeds the media of each sample, choosing the decoder by file
        extension (video files, audio files, ``.mcap``).

        Use this where ``compute_embeddings()`` does not apply: audio, the
        audio track of videos, video+audio together, and ``multimodal``
        collections such as MCAP. For grouped datasets, pass a slice view
        from ``select_group_slices()``.

        Args:
            samples: a :class:`fiftyone.core.collections.SampleCollection`
            embeddings_field (None): a sample field in which to store the
                embeddings. If None, the embeddings are returned
            modality (None): ``"video"``, ``"audio"`` or ``"video+audio"``.
                Defaults to the model's ``media_type``
            filepath_field ("filepath"): the field containing media paths
            skip_existing (False): skip samples whose ``embeddings_field`` is
                already populated
            skip_failures (True): log and store None for samples that fail
                instead of raising
            batch_size (64): number of samples written to the database at a
                time
            progress (None): whether to render a progress bar

        Returns:
            None if ``embeddings_field`` is given, else a list of arrays (None
            for failures)
        """
        modality = self._default_modality(modality)

        if samples.media_type == "group":
            raise ValueError(
                "Grouped collections are not supported directly; pass "
                "samples.select_group_slices(<slice>)"
            )

        view = samples
        if embeddings_field is not None:
            dataset = samples._dataset
            if not dataset.has_sample_field(embeddings_field):
                dataset.add_sample_field(embeddings_field, fof.VectorField)
            if skip_existing:
                view = view.exists(embeddings_field, False)

        ids, paths = view.values(["id", filepath_field])

        results = []
        pending = {}
        with fou.ProgressBar(total=len(ids), progress=progress) as pb:
            for sample_id, path in pb(zip(ids, paths)):
                try:
                    if path is None:
                        raise ValueError(f"Sample has no '{filepath_field}'")
                    embedding = self._embed_path(path, modality)
                except Exception as e:
                    if not skip_failures:
                        raise
                    logger.warning("Sample: %s\nError: %s\n", sample_id, e)
                    embedding = None

                if embeddings_field is None:
                    results.append(embedding)
                    continue

                pending[sample_id] = embedding
                if len(pending) >= batch_size:
                    samples.set_values(embeddings_field, pending, key_field="id")
                    pending = {}

        if embeddings_field is None:
            return results

        if pending:
            samples.set_values(embeddings_field, pending, key_field="id")

    def classify_dataset(
        self,
        samples,
        label_field,
        embeddings_field=None,
        modality=None,
        filepath_field="filepath",
        skip_failures=True,
        progress=None,
    ):
        """Zero-shot classifies each sample against ``classes``.

        Works on any collection :meth:`embed_dataset` supports. Pass an
        existing ``embeddings_field`` to reuse stored embeddings instead of
        decoding media again.

        Args:
            samples: a :class:`fiftyone.core.collections.SampleCollection`
            label_field: the sample field in which to store
                :class:`fiftyone.core.labels.Classification` labels
            embeddings_field (None): an optional field of stored embeddings
                from this model at the current ``embedding_dim``
            modality (None): see :meth:`embed_dataset`
            filepath_field ("filepath"): the field containing media paths
            skip_failures (True): see :meth:`embed_dataset`
            progress (None): whether to render a progress bar
        """
        # fail before decoding any media if classes is unset
        self._get_class_embeddings()

        if embeddings_field is not None:
            ids, vectors = samples.values(["id", embeddings_field])
        else:
            ids = samples.values("id")
            vectors = self.embed_dataset(
                samples,
                modality=modality,
                filepath_field=filepath_field,
                skip_failures=skip_failures,
                progress=progress,
            )

        keep = [(i, v) for i, v in zip(ids, vectors) if v is not None]
        if not keep:
            return

        matrix = np.stack([np.asarray(v, dtype=np.float32) for _, v in keep])
        if matrix.shape[1] != self.config.embedding_dim:
            raise ValueError(
                f"Stored embeddings have dimension {matrix.shape[1]} but "
                f"embedding_dim is {self.config.embedding_dim}"
            )

        labels = self._classify(matrix)
        samples.set_values(
            label_field,
            {i: label for (i, _), label in zip(keep, labels)},
            key_field="id",
        )

    def _check_budget(self, num_frames):
        tokens = self._budget_tokens(num_frames)
        if tokens > MAX_CONTEXT_TOKENS:
            raise ValueError(
                f"{num_frames} frames at {self.config.video_max_soft_tokens} "
                f"soft tokens need up to {tokens} tokens; the model supports "
                f"{MAX_CONTEXT_TOKENS}"
            )

    def _budget_tokens(self, num_frames=0, audio_seconds=0.0, text_tokens=0):
        per_frame = self.config.video_max_soft_tokens + 2
        audio_tokens = (
            math.ceil(audio_seconds * AUDIO_TOKENS_PER_SECOND) + 2
            if audio_seconds > 0
            else 0
        )
        return 2 + text_tokens + num_frames * per_frame + audio_tokens

    def fit_budget(self, num_frames=0, audio_seconds=0.0, text_tokens=0):
        """Shrinks a frame count and audio length to fit the context window.

        Video keeps priority: frames are reduced only if they alone overflow
        the window, then audio is truncated to the remaining room. A warning
        is logged when anything is reduced.

        Args:
            num_frames (0): requested video frames
            audio_seconds (0.0): requested audio duration (capped at
                ``max_audio_seconds``)
            text_tokens (0): tokens used by any accompanying text

        Returns:
            a ``(num_frames, audio_seconds)`` tuple
        """
        audio_seconds = min(audio_seconds, self.config.max_audio_seconds)

        per_frame = self.config.video_max_soft_tokens + 2
        frame_room = MAX_CONTEXT_TOKENS - self._budget_tokens(0, 0.0, text_tokens)
        fit_frames = max(1, frame_room // per_frame)
        if num_frames > fit_frames:
            logger.warning(
                "Reducing %d frames to %d to fit %d tokens",
                num_frames,
                fit_frames,
                MAX_CONTEXT_TOKENS,
            )
            num_frames = fit_frames

        if audio_seconds <= 0:
            return num_frames, audio_seconds

        if (
            self._budget_tokens(num_frames, audio_seconds, text_tokens)
            > MAX_CONTEXT_TOKENS
        ):
            # + 2 for the audio start/end tokens, - 1 for rounding up
            used = self._budget_tokens(num_frames, 0.0, text_tokens) + 2
            fit_seconds = (
                MAX_CONTEXT_TOKENS - used - 1
            ) / AUDIO_TOKENS_PER_SECOND
            if fit_seconds <= 0:
                raise ValueError(
                    "No room for audio in the context window; reduce "
                    "max_frames, video_max_soft_tokens or text length"
                )
            logger.warning(
                "Truncating audio from %.1fs to %.1fs to fit %d tokens "
                "alongside %d frames",
                audio_seconds,
                fit_seconds,
                MAX_CONTEXT_TOKENS,
                num_frames,
            )
            audio_seconds = fit_seconds

        return num_frames, audio_seconds

    ###########################################################################
    # Prompts and zero-shot
    ###########################################################################

    def embed_prompt(self, prompt):
        """Embeds a text query using ``prompt_name`` as prefix.

        Args:
            prompt: a string

        Returns:
            a ``(embedding_dim,)`` array
        """
        return self.embed_prompts([prompt])[0]

    def embed_prompts(self, prompts):
        """Embeds text queries using ``prompt_name`` as prefix.

        Args:
            prompts: a list of strings

        Returns:
            a ``(num_prompts, embedding_dim)`` array
        """
        return self.embed_text(list(prompts), prompt_name=self.config.prompt_name)

    def _get_class_embeddings(self):
        if not self.config.classes:
            raise ValueError(
                "Set model.classes to a list of class names before running "
                "zero-shot classification"
            )

        key = (
            tuple(self.config.classes),
            self.config.text_prompt,
            self.config.prompt_name,
            self.config.embedding_dim,
        )
        if key != self._class_embeddings_key:
            prompts = [
                self.config.text_prompt.format(c) for c in self.config.classes
            ]
            self._class_embeddings = self.embed_prompts(prompts)
            self._class_embeddings_key = key

        return self._class_embeddings

    def _classify(self, embeddings):
        class_embeddings = self._get_class_embeddings()
        logits = self.config.logit_scale * embeddings @ class_embeddings.T
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)

        labels = []
        for p in probs:
            idx = int(np.argmax(p))
            labels.append(
                fol.Classification(
                    label=self.config.classes[idx], confidence=float(p[idx])
                )
            )

        return labels


class EmbeddingGemma2ImageModel(
    EmbeddingGemma2Base,
    fom.Model,
    fom.EmbeddingsMixin,
    fom.PromptMixin,
    fom.SupportsGetItem,
    fom.TorchModelMixin,
):
    """EmbeddingGemma 2 for image datasets.

    Supports ``compute_embeddings``, text queries via ``compute_similarity``,
    and zero-shot classification via ``apply_model`` when ``classes`` is set.

    Args:
        config: an :class:`EmbeddingGemma2Config`
    """

    def __init__(self, config):
        fom.SupportsGetItem.__init__(self)
        EmbeddingGemma2Base.__init__(self, config)
        self._fields = {}
        self._mode = None

    @property
    def media_type(self):
        return "image"

    @property
    def mode(self):
        """The media type of the collection being processed.

        ``compute_embeddings()`` sets this while it runs; it is only used to
        warn when an image model is applied to a video collection.
        """
        return self._mode

    @mode.setter
    def mode(self, value):
        if value == "video":
            logger.warning(
                "Image model applied to a video collection: every frame gets "
                "its own embedding in a frame-level field. For one vector "
                "per video, load with media_type='video' (or "
                "media_type=dataset.media_type)"
            )
        self._mode = value

    @property
    def has_collate_fn(self):
        return True

    @property
    def needs_fields(self):
        return self._fields

    @needs_fields.setter
    def needs_fields(self, fields):
        self._fields = fields

    def build_get_item(self, field_mapping=None):
        return ImageGetItem(field_mapping=field_mapping, raw_inputs=True)

    def embed_all(self, args):
        return self.embed_images(list(args))

    def predict_all(self, args):
        return self._classify(self.embed_images(list(args)))


class EmbeddingGemma2VideoModel(
    EmbeddingGemma2Base,
    fom.Model,
    fom.EmbeddingsMixin,
    fom.PromptMixin,
):
    """EmbeddingGemma 2 for video datasets: one vector per video or clip.

    FiftyOne passes an ``FFmpegVideoReader`` per sample; only its path and
    frame range are used, and frames are decoded by timestamp with PyAV.
    Clip views are embedded over the clip's time window.

    ``embed()`` and ``predict()`` also accept a video or ``.mcap`` filepath,
    or a list of frames (PIL images or numpy HWC arrays).

    Args:
        config: an :class:`EmbeddingGemma2Config`
    """

    @property
    def media_type(self):
        return "video"

    def _to_frames(self, arg):
        if isinstance(arg, etav.VideoReader):
            first, last = etaf.parse_frame_ranges(arg.frames).limits
            # container frame counts can be wrong, so full videos use the
            # decoded duration instead of a window derived from them
            if first <= 1 and last >= arg.total_frame_count:
                return self.load_frames(arg.inpath)

            fps = arg.frame_rate
            return self.load_frames(
                arg.inpath, start=(first - 1) / fps, end=last / fps
            )

        if isinstance(arg, str):
            return self.load_frames(arg)

        return self._fit_frames(arg)

    def embed_all(self, args):
        return self.embed_frames([self._to_frames(arg) for arg in args])

    def predict_all(self, args):
        return self._classify(self.embed_all(args))


class EmbeddingGemma2AudioModel(
    EmbeddingGemma2Base,
    fom.Model,
    fom.EmbeddingsMixin,
    fom.PromptMixin,
):
    """EmbeddingGemma 2 for audio.

    FiftyOne's ``compute_embeddings()`` and ``apply_model()`` do not accept
    audio, so use :meth:`embed_dataset` and :meth:`classify_dataset`. Audio
    is read from audio files, the audio track of video files, or MCAP
    ``foxglove.RawAudio`` topics.

    ``embed()`` and ``predict()`` accept a filepath or a 16 kHz mono float32
    waveform.

    Args:
        config: an :class:`EmbeddingGemma2Config`
    """

    @property
    def media_type(self):
        return "audio"

    def embed_all(self, args):
        args = list(args)
        vectors = [None] * len(args)
        waves = []
        for i, arg in enumerate(args):
            if isinstance(arg, str):
                vectors[i] = self.embed_audio_file(arg)
            else:
                waves.append((i, arg))

        if waves:
            embedded = self.embed_audio([w for _, w in waves])
            for (i, _), e in zip(waves, embedded):
                vectors[i] = e

        return np.stack(vectors)

    def predict_all(self, args):
        return self._classify(self.embed_all(args))
