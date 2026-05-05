"""
Fundus image analysis: ResNet-18 encoder with GradCAM explanations.
"""
from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

log = logging.getLogger(__name__)

SEVERITY_LABELS = [
    "Normal",
    "Glaucoma Suspect",
    "Mild Glaucoma",
    "Moderate Glaucoma",
    "Severe Glaucoma",
]

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def _build_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


class FundusEncoder(nn.Module):
    """
    ResNet-18 backbone (ImageNet pretrained) + 5-class glaucoma severity head.
    GradCAM is computed via hooks on the final convolutional block (layer4).
    """

    def __init__(self, n_classes: int = 5) -> None:
        super().__init__()
        from torchvision import models

        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Everything up to (but not including) the global pool and FC:
        # output: [B, 512, 7, 7] for a 224×224 input
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )

        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None

        # Hooks on layer4 (self.features[-1])
        self.features[-1].register_forward_hook(self._fwd_hook)
        self.features[-1].register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self._activations = out   # keep in graph for backward

    def _bwd_hook(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)          # [B, 512, 7, 7]
        return self.head(self.pool(feat))  # [B, n_classes]

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return [B, 512] feature embedding without gradient."""
        with torch.no_grad():
            return self.pool(self.features(x)).flatten(1)

    def gradcam(
        self,
        x: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute GradCAM for the given input.
        Returns a [7, 7] float32 array normalised to [0, 1].
        """
        self.eval()
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = self(x_in)
            idx = class_idx if class_idx is not None else int(logits.argmax(-1).item())
            self.zero_grad()
            logits[0, idx].backward()

        if self._gradients is None or self._activations is None:
            return np.zeros((7, 7), dtype=np.float32)

        # Global average pool gradients → channel weights
        w = self._gradients.mean(dim=(2, 3), keepdim=True)          # [1, 512, 1, 1]
        cam = F.relu((w * self._activations.detach()).sum(dim=1))    # [1, 7, 7]
        cam = cam.squeeze().cpu().numpy()

        mn, mx = cam.min(), cam.max()
        if mx - mn > 1e-8:
            cam = (cam - mn) / (mx - mn)
        return cam.astype(np.float32)


def load_image_tensor(image_bytes: bytes, device: torch.device) -> torch.Tensor:
    """Decode raw image bytes → normalised [1, 3, 224, 224] tensor."""
    transform = _build_transform()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(img).unsqueeze(0).to(device)


def overlay_gradcam(cam: np.ndarray, image_bytes: bytes) -> str:
    """
    Blend the GradCAM heatmap onto the original image.
    Returns a base64-encoded PNG string.
    Pure PIL/numpy — no OpenCV dependency.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    img_np = np.array(img, dtype=np.float32)

    # Upsample 7×7 → 224×224
    cam_pil = Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    c = np.array(cam_pil, dtype=np.float32) / 255.0   # [224, 224] in [0, 1]

    # Jet colormap (piecewise linear approximation)
    r = np.clip(1.5 - np.abs(4.0 * c - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * c - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * c - 1.0), 0.0, 1.0)
    jet = np.stack([r, g, b], axis=-1) * 255.0

    overlay = np.uint8(0.55 * img_np + 0.45 * jet)
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def image_preview_b64(image_bytes: bytes) -> str:
    """Resize original image to 224×224 and return as base64 PNG."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def anatomical_findings(cam: np.ndarray) -> List[Tuple[str, float]]:
    """
    Map GradCAM activation regions to ophthalmic anatomical structures.
    cam: [7, 7] normalised to [0, 1].
    """
    regions: Dict[str, np.ndarray] = {
        "Optic disc / cup":   cam[2:5, 2:5],
        "Superior RNFL":      cam[0:2, :],
        "Inferior RNFL":      cam[5:,  :],
        "Nasal retina":       cam[:, 0:2],
        "Temporal retina":    cam[:, 5:],
        "Peripapillary rim":  np.concatenate([
            cam[1:6, 1:2].flatten(),
            cam[1:6, 5:6].flatten(),
            cam[1:2, 1:6].flatten(),
            cam[5:6, 1:6].flatten(),
        ]),
    }
    scored = [
        (name, round(float(patch.mean()), 3))
        for name, patch in regions.items()
        if patch.mean() > 0.02
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:5]
