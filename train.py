"""
Training Entry Point for the Broadcast NER System.

Usage examples
--------------
    # Train with default settings (CoNLL-2003 data, 4 epochs)
    python train.py

    # Standard training (no fine-tuning strategy)
    python train.py --epochs 8 --batch_size 16 --lr_encoder 2e-5

    # Two-phase discriminative fine-tuning (recommended)
    python train.py --fine_tune --freeze_epochs 2 --epochs 6

    # Train with stronger regularisation to reduce overfitting
    python train.py --label_smoothing 0.15 --dropout 0.4 --patience 4

    # GPU training with mixed precision
    python train.py --device cuda --amp --fine_tune

    # Full example
    python train.py --epochs 10 --batch_size 16 --lr_encoder 2e-5 \\
                    --lr_classifier 5e-4 --label_smoothing 0.1 \\
                    --fine_tune --freeze_epochs 3 --patience 4 \\
                    --device cuda --amp --seed 42
"""

import argparse
import json
import os
import sys
import time
import torch

# Add the project root to sys.path so all src.* imports resolve correctly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import NERConfig, ModelConfig, TrainingConfig, DataConfig, PathConfig
from src.data.ner_dataset import create_dataloaders
from src.model.ner_model import BroadcastNERModel
from src.training.trainer import NERTrainer
from transformers import BertTokenizerFast


# ==============================================================================
# Argument parsing
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the Broadcast NER Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model settings ────────────────────────────────────────────────────────
    parser.add_argument(
        "--model_name", type=str, default="bert-base-uncased",
        help="Pre-trained BERT model name or local path",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=128,
        help="Maximum sub-word token sequence length",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.3,
        help="Dropout probability on the BERT output (regularisation)",
    )
    parser.add_argument(
        "--no_crf", action="store_true",
        help="Disable the CRF layer and use argmax decoding instead",
    )

    # ── Learning rates ────────────────────────────────────────────────────────
    parser.add_argument(
        "--lr_encoder", type=float, default=2e-5,
        help="Learning rate for the BERT encoder layers",
    )
    parser.add_argument(
        "--lr_classifier", type=float, default=5e-4,
        help="Learning rate for the fresh classifier head",
    )

    # ── Regularisation ────────────────────────────────────────────────────────
    parser.add_argument(
        "--label_smoothing", type=float, default=0.1,
        help=(
            "Label smoothing epsilon (0 = disabled, 0.1 = recommended). "
            "Prevents over-confidence on training examples, reducing overfitting."
        ),
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01,
        help="L2 weight decay applied by AdamW",
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--epochs", type=int, default=4,
        help="Total number of training epochs",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Training and evaluation batch size",
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.1,
        help="Fraction of total training steps used for LR warm-up",
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0,
        help="Gradient clipping threshold",
    )
    parser.add_argument(
        "--gradient_accumulation", type=int, default=1,
        help="Accumulate gradients over N steps before each optimizer update",
    )
    parser.add_argument(
        "--scheduler", type=str, default="linear", choices=["linear", "cosine"],
        help="Learning rate schedule after the warm-up phase",
    )

    # ── Discriminative fine-tuning ────────────────────────────────────────────
    parser.add_argument(
        "--fine_tune", action="store_true", default=False,
        help=(
            "Enable two-phase discriminative fine-tuning: "
            "Phase 1 trains only the classifier head (encoder frozen), "
            "Phase 2 trains the full model with a lower encoder LR."
        ),
    )
    parser.add_argument(
        "--freeze_epochs", type=int, default=2,
        help="Epochs for Phase 1 of fine-tuning (classifier warm-up)",
    )

    # ── Runtime ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--device", type=str, default=None,
        help="Compute device: 'cuda' or 'cpu'. Auto-detected if not specified.",
    )
    parser.add_argument(
        "--amp", action="store_true",
        help="Enable Automatic Mixed Precision (GPU only)",
    )
    parser.add_argument(
        "--patience", type=int, default=3,
        help="Early stopping patience in epochs (0 to disable)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )

    # ── Dataset size controls ─────────────────────────────────────────────────
    parser.add_argument(
        "--full_data", action="store_true", default=False,
        help=(
            "Use the full CoNLL-2003 dataset (14,041 train / 3,250 val / 3,684 test). "
            "Overrides --train_subset / --val_subset / --test_subset. "
            "WARNING: takes ~5 hours on CPU."
        ),
    )
    parser.add_argument(
        "--train_subset", type=int, default=1500,
        help="Max training samples to use (default 1500 for quick runs). "
             "Ignored when --full_data is set.",
    )
    parser.add_argument(
        "--val_subset", type=int, default=200,
        help="Max validation samples. Ignored when --full_data is set.",
    )
    parser.add_argument(
        "--test_subset", type=int, default=200,
        help="Max test samples. Ignored when --full_data is set.",
    )

    return parser.parse_args()


# ==============================================================================
# Reproducibility
# ==============================================================================

def set_seed(seed: int):
    """Set all random seeds for reproducible training runs.

    This ensures that training results are deterministic when the same seed
    and hardware configuration are used.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False  # deterministic > speed


# ==============================================================================
# Configuration summary
# ==============================================================================

def print_config_summary(args: argparse.Namespace):
    """Print a formatted summary of the training configuration."""
    fine_tune_str = (
        f"Two-phase (freeze_epochs={args.freeze_epochs})"
        if args.fine_tune
        else "Standard (single-phase)"
    )
    print("\n" + "=" * 60)
    print("  Training Configuration")
    print("=" * 60)
    print(f"  Model          : {args.model_name}")
    print(f"  Max seq length : {args.max_seq_length}")
    print(f"  Dropout        : {args.dropout}")
    print(f"  Label smoothing: {args.label_smoothing}")
    print(f"  Epochs         : {args.epochs}")
    print(f"  Batch size     : {args.batch_size}")
    print(f"  LR (encoder)   : {args.lr_encoder}")
    print(f"  LR (classifier): {args.lr_classifier}")
    print(f"  Weight decay   : {args.weight_decay}")
    print(f"  Scheduler      : {args.scheduler}")
    print(f"  Warmup ratio   : {args.warmup_ratio}")
    print(f"  Fine-tuning    : {fine_tune_str}")
    print(f"  Patience       : {args.patience}")
    print(f"  AMP            : {args.amp}")
    print(f"  Seed           : {args.seed}")
    data_mode = "FULL CoNLL-2003" if args.full_data else f"Subset (train={args.train_subset}, val={args.val_subset}, test={args.test_subset})"
    print(f"  Dataset        : {data_mode}")
    print("=" * 60 + "\n")



# ==============================================================================
# Main
# ==============================================================================

def main():
    """Main training entry point."""
    args = parse_args()

    # Set random seeds before anything else for full reproducibility.
    set_seed(args.seed)

    print("\n" + "=" * 60)
    print("  Broadcast NER System — Training Pipeline")
    print("  Model: BERT-based Named Entity Recognition")
    print("  Labels: PER | ORG | LOC | MISC")
    print("=" * 60)

    print_config_summary(args)

    # ── Build typed configuration objects ────────────────────────────────────
    model_config = ModelConfig(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dropout_rate=args.dropout,
        use_crf=not args.no_crf,
    )

    training_config = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        encoder_learning_rate=args.lr_encoder,
        classifier_learning_rate=args.lr_classifier,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation,
        scheduler_type=args.scheduler,
        label_smoothing=args.label_smoothing,
        use_amp=args.amp,
        patience=args.patience,
    )

    # Resolve dataset subset limits.
    # --full_data overrides individual subset flags (use entire CoNLL-2003).
    if args.full_data:
        train_subset = None
        val_subset   = None
        test_subset  = None
        print("[DATA] Full dataset mode: all CoNLL-2003 splits will be used.")
    else:
        train_subset = args.train_subset
        val_subset   = args.val_subset
        test_subset  = args.test_subset

    data_config = DataConfig(
        max_seq_length=args.max_seq_length,
        data_subset_train=train_subset,
        data_subset_val=val_subset,
        data_subset_test=test_subset,
    )


    config = NERConfig(
        model=model_config,
        training=training_config,
        data=data_config,
    )

    # ── Load tokenizer ────────────────────────────────────────────────────────
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)

    # ── Create data loaders ───────────────────────────────────────────────────
    print("Preparing datasets ...")
    train_loader, val_loader, test_loader = create_dataloaders(
        tokenizer=tokenizer,
        config=data_config,
        batch_size=args.batch_size,
        num_workers=0,  # must be 0 on Windows due to multiprocessing constraints
    )

    # ── Build model ───────────────────────────────────────────────────────────
    print(f"Building model: {args.model_name}")
    model = BroadcastNERModel(
        config=model_config,
        label_smoothing=args.label_smoothing,
    )

    # ── Create trainer ────────────────────────────────────────────────────────
    trainer = NERTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=args.device,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    start_time = time.time()

    if args.fine_tune:
        print("Strategy: Discriminative fine-tuning (Phase 1 → Phase 2)")
        history = trainer.train_with_fine_tuning(freeze_epochs=args.freeze_epochs)
    else:
        print("Strategy: Standard single-phase training")
        history = trainer.train()

    elapsed = time.time() - start_time
    print(f"\nTotal training time: {elapsed:.1f}s  ({elapsed / 60:.1f} min)")

    # ── Evaluate on test set with the best saved model ────────────────────────
    print("\nLoading best model checkpoint for test evaluation ...")
    if os.path.exists(config.paths.best_model_path):
        best_model = BroadcastNERModel.load_model(
            config.paths.best_model_path,
            device=str(trainer.device),
        )
        trainer.model = best_model

    trainer.evaluate(test_loader)

    # ── Persist training history ──────────────────────────────────────────────
    os.makedirs(config.paths.output_dir, exist_ok=True)
    history_path = os.path.join(config.paths.output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to: {history_path}")

    print("\n" + "=" * 60)
    print("  Training complete. Model is ready for inference.")
    print("  Run predictions with:  python predict.py --demo")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
