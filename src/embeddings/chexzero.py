"""CheXzero (Tiu et al. 2022): OpenAI CLIP ViT-B/32 fine-tuned on MIMIC-CXR.

Not on HF. The authors publish state dicts on Google Drive, so the pin is the
checkpoint's sha256, and a download that does not match stops the run.
Preprocessing follows the original repo exactly, because the model was trained
on it: letterbox to 320, 0-255 scale, CXR-specific mean/std, then resize to 224.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from src.embeddings.base import l2_normalize, load_rgb

WEIGHTS_DIR = Path("data/cache/models/chexzero")
LETTERBOX = 320
INPUT_RESOLUTION = 224
MEAN, STD = 101.48761, 83.43944     # repo zero_shot.py, computed on MIMIC-CXR
CONTEXT_LENGTH = 77


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def verify_sha256(path: Path, expected: str | None) -> None:
    actual = sha256_of(path)
    if not expected:
        raise ValueError(
            f"{path} is not pinned. Set image_encoder.candidates.chexzero.revision "
            f"in configs/models.yaml to {actual} after checking the download."
        )
    if actual != expected:
        raise ValueError(f"{path}: sha256 mismatch, expected {expected}, got {actual}")


def ensure_checkpoint(candidate: dict, weights_dir: Path = WEIGHTS_DIR) -> Path:
    path = weights_dir / candidate["filename"]
    if not path.is_file():
        import gdown

        weights_dir.mkdir(parents=True, exist_ok=True)
        gdown.download(id=candidate["gdrive_file_id"], output=str(path), quiet=False)
    verify_sha256(path, candidate.get("revision"))
    return path


def chexzero_pixels(path: Path) -> np.ndarray:
    """Letterbox onto a black 320x320 greyscale canvas; [3, 320, 320], 0-255."""
    image = load_rgb(path)
    ratio = LETTERBOX / max(image.size)
    size = tuple(int(side * ratio) for side in image.size)
    image = image.resize(size, Image.LANCZOS)
    canvas = Image.new("L", (LETTERBOX, LETTERBOX))
    canvas.paste(image, ((LETTERBOX - size[0]) // 2, (LETTERBOX - size[1]) // 2))
    grey = np.asarray(canvas, dtype=np.float32)
    return np.repeat(grey[None], 3, axis=0)


class CheXzeroEncoder:
    name = "chexzero"

    def __init__(self, candidate: dict, device: str = "cpu"):
        import open_clip
        import torch
        from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize

        checkpoint = ensure_checkpoint(candidate)
        self.model_id = candidate["model_id"]
        self.revision = candidate["revision"]
        self.license = candidate["license"]
        self.device = device
        self._torch = torch
        self._open_clip = open_clip

        state = torch.load(checkpoint, map_location="cpu")
        self._model = (
            open_clip.model.build_model_from_openai_state_dict(state).float().to(device).eval()
        )
        # Normalize before resize, as the original repo does.
        self._transform = Compose([
            Normalize((MEAN,) * 3, (STD,) * 3),
            Resize(INPUT_RESOLUTION, interpolation=InterpolationMode.BICUBIC, antialias=False),
        ])
        self.weights_bytes = checkpoint.stat().st_size
        self.preprocessing = {
            "letterbox": LETTERBOX,
            "letterbox_resample": "LANCZOS",
            "scale": "0-255",
            "mean": MEAN,
            "std": STD,
            "resize": f"{INPUT_RESOLUTION} bicubic, antialias=False, after normalize",
            "decoder": "PIL (original repo decodes with cv2)",
            "context_length": CONTEXT_LENGTH,
            "checkpoint": candidate["filename"],
        }
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.encode_texts(["chest x-ray"]).shape[1])
        return self._dim

    def encode_images(self, paths: Sequence[Path]) -> np.ndarray:
        batch = self._torch.stack(
            [self._transform(self._torch.from_numpy(chexzero_pixels(p))) for p in paths]
        ).to(self.device)
        with self._torch.inference_mode():
            return l2_normalize(self._model.encode_image(batch).float().cpu().numpy())

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        tokens = self._open_clip.tokenize(list(texts), context_length=CONTEXT_LENGTH)
        with self._torch.inference_mode():
            return l2_normalize(
                self._model.encode_text(tokens.to(self.device)).float().cpu().numpy()
            )
