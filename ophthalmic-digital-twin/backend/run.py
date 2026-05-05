"""
Entry point for training, serving, and evaluation.

Usage:
  python run.py --mode train    # full training pipeline
  python run.py --mode serve    # start FastAPI on :8000
  python run.py --mode evaluate # load best model, print all metrics
  python run.py --mode train --device cuda
  python run.py --mode serve --port 8000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure backend/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("run")


def _resolve_device(device_arg: str | None):
    import torch
    if device_arg is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def run_train(args: argparse.Namespace) -> None:
    import torch
    from core.dataset import load_data
    from core.model import build_model
    from core.trainer import train, evaluate

    device = _resolve_device(args.device)
    log.info("=== TRAINING MODE | device=%s ===", device)

    csv_path = args.csv or str(Path(__file__).resolve().parent.parent / "data" / "full_df.csv")
    if not Path(csv_path).exists():
        log.error("Dataset not found at %s. Place your CSV at data/full_df.csv", csv_path)
        sys.exit(1)

    train_loader, val_loader, test_loader, feature_metadata = load_data(
        csv_path=csv_path,
        batch_size=64,
    )

    model = build_model(feature_metadata)
    log.info("Model: %s | params=%d", type(model).__name__,
             sum(p.numel() for p in model.parameters()))

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        feature_metadata=feature_metadata,
        n_epochs=100,
        device=device,
    )

    log.info("=== EVALUATION on TEST SET ===")
    metrics = evaluate(model, test_loader, feature_metadata, device=device)
    for k, v in metrics.items():
        log.info("  %s: %.4f", k, v)


def run_serve(args: argparse.Namespace) -> None:
    import uvicorn
    port = getattr(args, "port", 8000)
    log.info("=== SERVE MODE | port=%d ===", port)
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )


def run_evaluate(args: argparse.Namespace) -> None:
    import torch
    from core.dataset import load_data
    from core.model import build_model
    from core.trainer import load_best_model, evaluate
    from core.uncertainty import calibration_metrics

    device = _resolve_device(args.device)
    log.info("=== EVALUATE MODE | device=%s ===", device)

    csv_path = args.csv or str(Path(__file__).resolve().parent.parent / "data" / "full_df.csv")
    _, val_loader, test_loader, feature_metadata = load_data(csv_path=csv_path, batch_size=64)

    model = build_model(feature_metadata)
    model, saved_meta = load_best_model(model, device=device)

    log.info("=== Test Metrics ===")
    metrics = evaluate(model, test_loader, feature_metadata, device=device)
    for k, v in metrics.items():
        log.info("  %s: %.4f", k, v)

    if feature_metadata["task"] == "classification":
        log.info("=== Calibration (ECE) on Validation Set ===")
        cal = calibration_metrics(model, val_loader, device=device)
        log.info("  ECE: %.4f", cal["ece"])
        log.info("  Reliability diagram points:")
        for conf, acc in cal["reliability_diagram"]:
            log.info("    conf=%.3f  acc=%.3f", conf, acc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ophthalmic Digital Twin runner")
    parser.add_argument(
        "--mode", choices=["train", "serve", "evaluate"], required=True,
        help="Operating mode."
    )
    parser.add_argument("--device", default=None, help="Force device: cuda or cpu")
    parser.add_argument("--csv", default=None, help="Path to full_df.csv")
    parser.add_argument("--port", type=int, default=8000, help="Port for serve mode")
    args = parser.parse_args()

    if args.mode == "train":
        run_train(args)
    elif args.mode == "serve":
        run_serve(args)
    elif args.mode == "evaluate":
        run_evaluate(args)


if __name__ == "__main__":
    main()
