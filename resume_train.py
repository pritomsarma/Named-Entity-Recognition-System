"""
Resume Training from a Saved Checkpoint.

Use this script after an interrupted training run or when you want to
continue fine-tuning a saved model for additional epochs.

Usage
-----
    # Resume with default settings (5 more epochs, lower LR)
    python resume_train.py

    # Resume on the full CoNLL-2003 dataset
    python resume_train.py --full_data

    # Resume with custom settings
    python resume_train.py --epochs 3 --lr_encoder 5e-6 --lr_classifier 1e-4
"""

import os
import sys
import time
import json
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import NERConfig, TrainingConfig, DataConfig
from src.data.ner_dataset import create_dataloaders
from src.model.ner_model import BroadcastNERModel
from src.training.trainer import NERTrainer
from transformers import BertTokenizerFast


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for resume training."""
    parser = argparse.ArgumentParser(
        description="Resume NER training from a saved checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_model.pt",
        help="Path to the model checkpoint to resume from",
    )
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Number of additional epochs to train",
    )
    parser.add_argument(
        "--lr_encoder", type=float, default=1e-5,
        help="Encoder learning rate (lower than initial training to avoid forgetting)",
    )
    parser.add_argument(
        "--lr_classifier", type=float, default=1e-4,
        help="Classifier head learning rate",
    )
    parser.add_argument(
        "--patience", type=int, default=3,
        help="Early stopping patience in epochs",
    )
    parser.add_argument(
        "--full_data", action="store_true", default=False,
        help="Use the full CoNLL-2003 dataset instead of the default subset",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Compute device: 'cuda' or 'cpu'",
    )
    return parser.parse_args()


def main():
    """Main resume-training entry point."""
    args = parse_args()

    print("\n" + "=" * 60)
    print("  Broadcast NER — Resuming Training")
    print("=" * 60)
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  Extra epochs: {args.epochs}")
    print(f"  LR (encoder): {args.lr_encoder}")
    print(f"  LR (classif): {args.lr_classifier}")
    print(f"  Patience    : {args.patience}")
    print(f"  Full data   : {args.full_data}")
    print("=" * 60 + "\n")

    # ── Build config ─────────────────────────────────────────────────────────
    config = NERConfig()

    # Override training settings for the resume run.
    # Note: use discriminative LRs (encoder_learning_rate / classifier_learning_rate),
    # NOT the removed `learning_rate` field.
    config.training.num_epochs               = args.epochs
    config.training.encoder_learning_rate    = args.lr_encoder
    config.training.classifier_learning_rate = args.lr_classifier
    config.training.patience                 = args.patience

    # Dataset subset control.
    if args.full_data:
        config.data.data_subset_train = None
        config.data.data_subset_val   = None
        config.data.data_subset_test  = None
        print("[DATA] Using full CoNLL-2003 dataset.")
    # else: keep the DataConfig defaults (1500 / 200 / 200)

    # ── Load tokenizer & data ─────────────────────────────────────────────────
    print(f"Loading tokenizer: {config.model.model_name}")
    tokenizer = BertTokenizerFast.from_pretrained(config.model.model_name)

    print("Preparing data loaders ...")
    train_loader, val_loader, test_loader = create_dataloaders(
        tokenizer=tokenizer,
        config=config.data,
        batch_size=args.batch_size,
        num_workers=0,  # 0 required on Windows
    )

    # ── Load checkpoint ───────────────────────────────────────────────────────
    if not os.path.exists(args.checkpoint):
        print(f"\n[ERROR] Checkpoint not found: {args.checkpoint}")
        print("  Train a model first with:  python train.py")
        sys.exit(1)

    print(f"Loading model from: {args.checkpoint}")
    model = BroadcastNERModel.load_model(args.checkpoint, device=args.device)

    # ── Create trainer and resume ─────────────────────────────────────────────
    trainer = NERTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=args.device,
    )

    start_time = time.time()
    history    = trainer.train()
    elapsed    = time.time() - start_time

    print(f"\nResume training time: {elapsed:.1f}s  ({elapsed / 60:.1f} min)")

    # ── Save history ──────────────────────────────────────────────────────────
    os.makedirs(config.paths.output_dir, exist_ok=True)
    history_path = os.path.join(config.paths.output_dir, "resume_training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[SAVE] Resume history saved to: {history_path}")

    # ── Evaluate on test set ──────────────────────────────────────────────────
    print("\n[EVAL] Running evaluation on test set ...")
    trainer.evaluate(test_loader)


if __name__ == "__main__":
    main()
