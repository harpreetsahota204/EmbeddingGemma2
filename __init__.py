"""EmbeddingGemma 2 remote zoo source for FiftyOne.

One manifest entry; ``load_zoo_model("google/embeddinggemma-2",
media_type=...)`` picks the image, video or audio model class.
"""

import logging

from huggingface_hub import snapshot_download

from .zoo import (
    DEFAULT_MODEL_PATH,
    VALID_MEDIA_TYPES,
    EmbeddingGemma2Config,
    EmbeddingGemma2ImageModel,
)

logger = logging.getLogger(__name__)


def download_model(model_name, model_path):
    """Downloads the model weights from the Hugging Face Hub.

    Safe to call repeatedly: files already present in ``model_path`` are not
    downloaded again.

    Args:
        model_name: the ``base_name`` from the manifest
        model_path: the directory to download to
    """
    snapshot_download(repo_id=model_name, local_dir=model_path)


def load_model(model_name=None, model_path=None, **kwargs):
    """Loads an EmbeddingGemma 2 model.

    Args:
        model_name: the ``base_name`` from the manifest
        model_path: a local directory or Hugging Face repo ID. Defaults to
            ``google/embeddinggemma-2``
        **kwargs: config parameters, including ``media_type`` (``"image"``,
            ``"video"`` or ``"audio"``; default ``"image"``)

    Returns:
        a :class:`fiftyone.core.models.Model`
    """
    config_dict = {"model_path": model_path or DEFAULT_MODEL_PATH}
    config_dict.update(kwargs)

    media_type = config_dict.get("media_type", "image")
    if media_type not in VALID_MEDIA_TYPES:
        raise ValueError(
            f"media_type must be one of {VALID_MEDIA_TYPES}, got '{media_type}'"
        )

    config = EmbeddingGemma2Config(config_dict)

    if media_type == "image":
        return EmbeddingGemma2ImageModel(config)

    raise NotImplementedError(f"media_type='{media_type}' is not built yet")
