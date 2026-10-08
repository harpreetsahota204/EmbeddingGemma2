"""EmbeddingGemma 2 models for FiftyOne.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

# Worker-pickle constraint: nothing in this module may be pickled across
# DataLoader workers. Use ImageGetItem and the inherited collate_fn only.

import logging
import math

import numpy as np
from PIL import Image
import torch
from transformers import AutoConfig, AutoModel, AutoProcessor

import fiftyone.core.labels as fol
import fiftyone.core.models as fom
import fiftyone.utils.torch as fout
from fiftyone.utils.torch import ImageGetItem

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
        classes (None): class names for zero-shot classification
        text_prompt ("a photo of a {}"): template for class prompts
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

        _validate_classes(self.classes)

        self.text_prompt = self.parse_string(
            d, "text_prompt", default="a photo of a {}"
        )
        _validate_text_prompt(self.text_prompt)

        self.logit_scale = self.parse_number(d, "logit_scale", default=100)
        _validate_positive("logit_scale", self.logit_scale)


class EmbeddingGemma2Base(object):
    """Media-agnostic EmbeddingGemma 2 core.

    Owns the weights, processor, device and dtype. All ``embed_*`` methods
    return float32, L2-normalized arrays of shape
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
            images: a list of PIL images, numpy HWC arrays or CHW tensors

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
            self._check_budget(num_frames=len(frames))

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

    def _check_budget(self, num_frames=0, audio_seconds=0.0, text_tokens=0):
        tokens = self._budget_tokens(num_frames, audio_seconds, text_tokens)
        if tokens > MAX_CONTEXT_TOKENS:
            raise ValueError(
                f"{num_frames} frames at {self.config.video_max_soft_tokens} "
                f"soft tokens + {audio_seconds:.1f}s audio needs up to "
                f"{tokens} tokens; the model supports {MAX_CONTEXT_TOKENS}"
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

        Frames are reduced first (to a minimum of 1), then audio is
        truncated. A warning is logged when anything is reduced.

        Args:
            num_frames (0): requested video frames
            audio_seconds (0.0): requested audio duration
            text_tokens (0): tokens used by any accompanying text

        Returns:
            a ``(num_frames, audio_seconds)`` tuple
        """
        audio_seconds = min(audio_seconds, self.config.max_audio_seconds)
        if (
            self._budget_tokens(num_frames, audio_seconds, text_tokens)
            <= MAX_CONTEXT_TOKENS
        ):
            return num_frames, audio_seconds

        per_frame = self.config.video_max_soft_tokens + 2
        fixed = self._budget_tokens(0, audio_seconds, text_tokens)
        fit_frames = max(1, (MAX_CONTEXT_TOKENS - fixed) // per_frame)
        if num_frames > fit_frames:
            logger.warning(
                "Reducing %d frames to %d to fit %d tokens",
                num_frames,
                fit_frames,
                MAX_CONTEXT_TOKENS,
            )
            num_frames = fit_frames

        if (
            self._budget_tokens(num_frames, audio_seconds, text_tokens)
            > MAX_CONTEXT_TOKENS
        ):
            fixed = self._budget_tokens(num_frames, 0.0, text_tokens) + 2
            fit_seconds = (
                MAX_CONTEXT_TOKENS - fixed - 1
            ) / AUDIO_TOKENS_PER_SECOND
            if fit_seconds <= 0:
                raise ValueError(
                    "No room for audio in the context window; reduce "
                    "video_max_soft_tokens or text length"
                )
            logger.warning(
                "Truncating audio from %.1fs to %.1fs to fit %d tokens",
                audio_seconds,
                fit_seconds,
                MAX_CONTEXT_TOKENS,
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
        self._preprocess = False
        self._fields = {}

    @property
    def media_type(self):
        return "image"

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

    def embed(self, arg):
        return self.embed_all([arg])[0]

    def embed_all(self, args):
        return self.embed_images(list(args))

    def predict(self, arg):
        return self.predict_all([arg])[0]

    def predict_all(self, args):
        return self._classify(self.embed_images(list(args)))
