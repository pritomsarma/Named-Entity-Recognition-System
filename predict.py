"""
Inference Entry Point for the Broadcast NER System.

Usage:
    # Demo mode with sample broadcast texts
    python predict.py --demo

    # Predict entities in a custom text
    python predict.py --text "Anderson Cooper reported from New York on CNN on January 15th."

    # Predict from a file (one sentence per line)
    python predict.py --file transcripts.txt

    # Run analytics on multiple texts
    python predict.py --demo --analytics

    # Use a specific model checkpoint
    python predict.py --model_path checkpoints/best_model.pt --text "Your text here"
"""

import argparse
import os
import sys
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import NERConfig, ModelConfig
from src.model.ner_model import BroadcastNERModel
from src.inference.predictor import BroadcastNERPredictor
from src.inference.analytics import BroadcastAnalytics


# ──────────────────────────────────────────────────────────────────────────────
# Sample broadcasting texts for demo mode
# ──────────────────────────────────────────────────────────────────────────────
DEMO_TEXTS = [
    "Anderson Cooper reported live from Washington D.C. on CNN on January 15th about the latest policy changes.",
    "According to BBC World Service, Rachel Maddow will anchor the special coverage from London starting February 2024.",
    "Breaking news from Tokyo: Reuters correspondent Lester Holt reports that officials have confirmed the trade agreement.",
    "Oprah Winfrey joined NBC News as a senior correspondent based in Los Angeles starting March 3rd, 2025.",
    "The broadcast aired on Monday featured an exclusive interview with Sundar Pichai in San Francisco.",
    "Fox News announced on Tuesday morning that Jake Tapper would lead the new prime-time show from New York.",
    "In Berlin, Christiane Amanpour covered the NATO summit for Al Jazeera on September 2024.",
    "Sources at The Washington Post confirmed that David Muir traveled to Moscow for the July 4th special report.",
    "Elon Musk delivered the keynote at the Bloomberg conference in Dubai on October 31st.",
    "The Wednesday evening edition of the NPR broadcast featured Wolf Blitzer reporting from Capitol Hill.",
    "Google has expanded its bureau in Silicon Valley, appointing Tim Cook as the new bureau chief.",
    "Viewers watched Savannah Guthrie present the breaking news segment from the White House on last Thursday.",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run NER prediction on broadcast text",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--text", type=str, default=None,
        help="Text string to predict entities for",
    )
    input_group.add_argument(
        "--file", type=str, default=None,
        help="Path to a text file (one sentence per line)",
    )
    input_group.add_argument(
        "--demo", action="store_true",
        help="Run demo with sample broadcast texts",
    )

    # Model options
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Path to trained model checkpoint. If not specified, uses an untrained model for demo.",
    )
    parser.add_argument(
        "--model_name", type=str, default="bert-base-uncased",
        help="Pre-trained BERT model name (for tokenizer)",
    )

    # Output options
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save prediction results as JSON",
    )
    parser.add_argument(
        "--analytics", action="store_true",
        help="Run broadcasting analytics on the predictions",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (cuda/cpu). Auto-detects if not specified.",
    )

    return parser.parse_args()


def print_header():
    """Print the application header."""
    print("\n--- Broadcast NER System: Inference Engine ---")
    print("Model: BERT-based Named Entity Recognition")
    print("Entities: Person | Organization | Location | Miscellaneous")
    print("-" * 46 + "\n")


def run_prediction(predictor: BroadcastNERPredictor, texts: list, run_analytics: bool = False):
    """Run predictions on a list of texts and display results.

    Args:
        predictor: BroadcastNERPredictor instance.
        texts: List of text strings.
        run_analytics: Whether to run broadcasting analytics.
    """
    all_results = []

    print(f"\nProcessing {len(texts)} text(s)...\n")
    print("-" * 50)

    for i, text in enumerate(texts, 1):
        result = predictor.predict(text)
        all_results.append(result)

        print(f"\n─── Sentence {i} {'─' * (55 - len(str(i)))}")
        print(f"  Text: {text}")
        print(f"\n  Tokens + Labels:")

        # Display tokens with their labels (colour-coded)
        for token, label in zip(result.tokens, result.labels):
            if label != "O":
                print(f"    {token:<25} → {label}")

        print(f"\n  Extracted Entities:")
        print(result.format_entities())
        print()

    print("-" * 50)
    print(f"\nProcessed {len(texts)} sentences, "
          f"extracted {sum(len(r.entities) for r in all_results)} entities total.\n")

    # Run analytics if requested
    if run_analytics:
        print("\nRunning broadcasting analytics...\n")
        analytics = BroadcastAnalytics()
        report = analytics.process_predictions(all_results)
        print(BroadcastAnalytics.format_report(report))

    return all_results


def main():
    """Main inference entry point."""
    args = parse_args()
    print_header()

    # ── Determine input texts ──
    if args.text:
        texts = [args.text]
    elif args.file:
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(texts)} sentences from {args.file}")
    elif args.demo:
        texts = DEMO_TEXTS
        print("Using demo broadcast texts")
    else:
        print("No input specified. Use --text, --file, or --demo.")
        print("Running demo mode by default.\n")
        texts = DEMO_TEXTS

    # ── Load predictor ──
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading trained model from: {args.model_path}")
        predictor = BroadcastNERPredictor(
            model_path=args.model_path,
            model_name=args.model_name,
            device=args.device,
        )
    else:
        if args.model_path:
            print(f"Warning: Checkpoint not found at: {args.model_path}")
        print("Using untrained BERT model (predictions will be random).")
        print("Train first with: python train.py\n")

        model_config = ModelConfig(model_name=args.model_name)
        model = BroadcastNERModel(config=model_config)
        predictor = BroadcastNERPredictor(
            model=model,
            model_name=args.model_name,
            device=args.device,
        )

    # ── Run predictions ──
    results = run_prediction(predictor, texts, run_analytics=args.analytics)

    # ── Save results if requested ──
    if args.output:
        output_data = [r.to_dict() for r in results]
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
