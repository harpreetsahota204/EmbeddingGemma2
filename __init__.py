"""EmbeddingGemma 2 remote zoo source for FiftyOne.

One manifest entry; ``load_zoo_model("google/embeddinggemma-2",
media_type=...)`` picks the image, video or audio model class.
"""

from huggingface_hub import snapshot_download

from .zoo import (
    DEFAULT_MODEL_PATH,
    EmbeddingGemma2AudioModel,
    EmbeddingGemma2Config,
    EmbeddingGemma2ImageModel,
    EmbeddingGemma2VideoModel,
)

_MODEL_CLASSES = {
    "image": EmbeddingGemma2ImageModel,
    "video": EmbeddingGemma2VideoModel,
    "audio": EmbeddingGemma2AudioModel,
}


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

    config = EmbeddingGemma2Config(config_dict)
    return _MODEL_CLASSES[config.media_type](config)
