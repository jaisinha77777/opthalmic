"""
Fundus image analysis: ResNet-18 encoder with GradCAM explanations.

Clinically honest task definition: a single color fundus photograph reveals optic-nerve
STRUCTURE, not visual-field FUNCTION. So this model does NOT attempt the visual-field-based
5-stage severity grade. Instead it predicts what a photo can actually support:

  1. Referable glaucoma  (binary): should this eye be referred to an ophthalmologist?
  2. Vertical cup-to-disc ratio (CDR): the key structural measurement (regression).

Trained weights are produced by scripts/train_fundus.py on a real public dataset
(REFUGE / G1020 / RIM-ONE). Without trained weights the model reports that it is
uncalibrated rather than emitting confident noise.
"""
from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

log = logging.getLogger(__name__)

# Binary referral decision (what a fundus photo can support).
REFERRAL_LABELS = ["No referable glaucoma", "Referable glaucoma"]

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def _build_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


class FundusEncoder(nn.Module):
    """
    ResNet-18 backbone (ImageNet pretrained) with two heads:
      - referral head : Linear(512 -> 2)   referable glaucoma (binary)
      - cdr head      : Linear(512 -> 1) + Sigmoid   vertical cup-disc ratio in [0,1]
    GradCAM is computed via hooks on the final convolutional block (layer4).
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        from torchvision import models

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet18(weights=weights)

        # Everything up to (but not including) global pool + FC -> [B, 512, 7, 7]
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.referral_head = nn.Linear(256, 2)
        self.cdr_head = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())

        self.is_calibrated: bool = False  # set True once real weights are loaded

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self.features[-1].register_forward_hook(self._fwd_hook)
        self.features[-1].register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self._activations = out

    def _bwd_hook(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.shared(self.pool(self.features(x)))   # [B, 256]
        return {
            "referral_logits": self.referral_head(feat),   # [B, 2]
            "cdr": self.cdr_head(feat).squeeze(-1),         # [B]
        }

    def load_weights(self, path: str | Path, device: torch.device) -> bool:
        """Load fine-tuned weights produced by train_fundus.py. Returns success flag."""
        path = Path(path)
        if not path.exists():
            log.warning("No fundus weights at %s -- model is UNCALIBRATED.", path)
            return False
        ckpt = torch.load(path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        self.load_state_dict(state)
        self.is_calibrated = True
        log.info("Loaded calibrated fundus weights from %s", path)
        return True

    def gradcam(self, x: torch.Tensor, target: str = "referral") -> np.ndarray:
        """GradCAM for the referable-glaucoma logit. Returns [7,7] in [0,1]."""
        self.eval()
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            out = self(x_in)
            score = out["referral_logits"][0, 1] if target == "referral" else out["cdr"][0]
            self.zero_grad()
            score.backward()

        if self._gradients is None or self._activations is None:
            return np.zeros((7, 7), dtype=np.float32)

        w = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self._activations.detach()).sum(dim=1)).squeeze().cpu().numpy()
        mn, mx = cam.min(), cam.max()
        if mx - mn > 1e-8:
            cam = (cam - mn) / (mx - mn)
        return cam.astype(np.float32)


def load_image_tensor(image_bytes: bytes, device: torch.device) -> torch.Tensor:
    transform = _build_transform()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(img).unsqueeze(0).to(device)


def overlay_gradcam(cam: np.ndarray, image_bytes: bytes) -> str:
    """Blend the GradCAM heatmap onto the original image -> base64 PNG."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    img_np = np.array(img, dtype=np.float32)

    cam_pil = Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    c = np.array(cam_pil, dtype=np.float32) / 255.0

    r = np.clip(1.5 - np.abs(4.0 * c - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * c - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * c - 1.0), 0.0, 1.0)
    jet = np.stack([r, g, b], axis=-1) * 255.0

    overlay = np.uint8(0.55 * img_np + 0.45 * jet)
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def image_preview_b64(image_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def anatomical_findings(cam: np.ndarray) -> List[Tuple[str, float]]:
    """Map GradCAM activation regions to ophthalmic structures (cam: [7,7] in [0,1])."""
    regions: Dict[str, np.ndarray] = {
        "Optic disc / cup": cam[2:5, 2:5],
        "Superior RNFL": cam[0:2, :],
        "Inferior RNFL": cam[5:, :],
        "Nasal retina": cam[:, 0:2],
        "Temporal retina": cam[:, 5:],
        "Peripapillary rim": np.concatenate([
            cam[1:6, 1:2].flatten(), cam[1:6, 5:6].flatten(),
            cam[1:2, 1:6].flatten(), cam[5:6, 1:6].flatten(),
        ]),
    }
    scored = [(name, round(float(p.mean()), 3)) for name, p in regions.items() if p.mean() > 0.02]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:5]
