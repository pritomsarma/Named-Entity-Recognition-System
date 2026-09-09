"""
Central Configuration for the Broadcast NER System.

All hyperparameters, label mappings, file paths, and model settings are kept
here as typed dataclasses so that nothing is hard-coded elsewhere in the project.

Sections
--------
1. BIO Label Scheme  — the 9 tags the model predicts
2. ModelConfig       — BERT encoder and classifier settings
3. TrainingConfig    — optimiser, scheduler, regularisation, and loop settings
4. DataConfig        — sequence length, dataset split sizes, and subset limits
5. PathConfig        — directory and file paths derived from the project root
6. NERConfig         — master config that bundles all of the above
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os


# ==============================================================================
# 1. BIO Tag Scheme
# ==============================================================================
# We use the standard BIO (Begin / Inside / Outside) encoding with four entity
# types that are common in broadcast-media analytics:
#   PER   — person names (anchors, reporters, public figures)
#   ORG   — organisations (networks, companies, institutions)
#   LOC   — locations (cities, countries, landmarks)
#   MISC  — miscellaneous named entities (events, dates, product names)

NER_LABELS: List[str] = [
    "O",       # Outside — token is not part of any named entity
    "B-PER",   # Begin of a Person entity
    "I-PER",   # Inside a Person entity (continuation)
    "B-ORG",   # Begin of an Organisation entity
    "I-ORG",   # Inside an Organisation entity
    "B-LOC",   # Begin of a Location entity
    "I-LOC",   # Inside a Location entity
    "B-MISC",  # Begin of a Miscellaneous entity
    "I-MISC",  # Inside a Miscellaneous entity
]

# Convenient lookup dictionaries built from the label list above.
LABEL2ID: Dict[str, int] = {label: idx for idx, label in enumerate(NER_LABELS)}
ID2LABEL: Dict[int, str] = {idx: label for idx, label in enumerate(NER_LABELS)}
NUM_LABELS: int = len(NER_LABELS)

# Tokens that should not contribute to the loss (padding, [CLS], [SEP], and
# the continuation sub-words of multi-piece words) are assigned this sentinel.
IGNORE_LABEL_ID: int = -100


# ==============================================================================
# 2. ModelConfig
# ==============================================================================

@dataclass
class ModelConfig:
    """Configuration for the BERT-based NER model architecture.

    The model is:
        Input tokens
            → BERT encoder  (contextual embeddings)
            → Dropout       (regularisation)
            → Linear layer  (project to num_labels)
            → [Optional CRF] (structured prediction)
    """

    # Hugging Face model identifier or local path to a pre-trained BERT model.
    model_name: str = "bert-base-uncased"

    # Total number of NER output classes (should match len(NER_LABELS)).
    num_labels: int = NUM_LABELS

    # Hidden dimension of the BERT encoder (768 for bert-base-*).
    # This is read back from the loaded model's own config after init.
    hidden_size: int = 768

    # Dropout probability applied to the encoder output before the classifier.
    # Higher values → more regularisation, helps prevent overfitting.
    dropout_rate: float = 0.3

    # When True, the BERT encoder parameters are frozen at the start of
    # training and only the classifier head is updated.
    freeze_encoder: bool = False

    # Maximum sub-word token sequence length fed to the model.
    max_seq_length: int = 128

    # When True, a CRF layer is added on top of the linear classifier to
    # enforce valid BIO tag sequences during decoding.
    use_crf: bool = True


# ==============================================================================
# 3. TrainingConfig
# ==============================================================================

@dataclass
class TrainingConfig:
    """Configuration for the training loop and all regularisation settings.

    Anti-overfitting levers
    -----------------------
    - dropout_rate     : set in ModelConfig (default 0.3)
    - label_smoothing  : softens one-hot targets so the model does not become
                         over-confident; especially important on small datasets.
    - weight_decay     : L2 penalty on the AdamW optimiser.
    - patience         : early-stopping stops training when validation F1 has
                         not improved for this many consecutive epochs.
    - warmup_ratio     : the learning rate ramps up slowly at the start,
                         preventing large gradient updates on a cold model.
    """

    # --- Discriminative learning rates ---
    # The BERT encoder uses a *lower* learning rate than the fresh classifier
    # head. This technique (from ULMFiT) prevents catastrophic forgetting of
    # the encoder's pre-trained knowledge.
    encoder_learning_rate: float = 2e-5
    classifier_learning_rate: float = 5e-4

    # --- AdamW optimiser settings ---
    weight_decay: float = 0.01   # L2 regularisation on non-bias parameters
    adam_epsilon: float = 1e-8   # numerical stability constant for Adam

    # --- Training loop ---
    num_epochs: int = 4
    batch_size: int = 16
    gradient_accumulation_steps: int = 1  # simulate larger batch sizes
    max_grad_norm: float = 1.0            # gradient clipping threshold

    # --- Learning rate schedule ---
    warmup_ratio: float = 0.1     # fraction of total steps used for LR warmup
    scheduler_type: str = "linear"  # "linear" or "cosine" decay after warmup

    # --- Regularisation ---
    # Label smoothing: instead of training against hard 0/1 targets, the true
    # class gets probability (1 - ε) and the remaining ε is spread uniformly
    # across all other classes.  This discourages over-confidence.
    label_smoothing: float = 0.1

    # --- Mixed precision ---
    use_amp: bool = True  # Automatic Mixed Precision (GPU only)

    # --- Checkpointing ---
    save_best_model: bool = True

    # --- Early stopping ---
    # Training stops when validation F1 has not improved for `patience` epochs.
    # Set to 0 to disable early stopping.
    patience: int = 3

    # --- Logging ---
    log_interval: int = 50   # log step-level loss every N optimizer steps
    use_tensorboard: bool = True


# ==============================================================================
# 4. DataConfig
# ==============================================================================

@dataclass
class DataConfig:
    """Configuration for data loading, preprocessing, and dataset splits.

    Dataset subsets
    ---------------
    We cap the CoNLL-2003 splits to keep training time reasonable during
    development. Increase these numbers for a full production training run.
    The full CoNLL-2003 dataset has ~14,000 training, ~3,250 val, ~3,680 test.
    """

    # Sub-word token sequence length passed to the BERT tokenizer.
    max_seq_length: int = 128

    # Maximum number of samples to use from each split.
    # Set to None to use the entire split.
    data_subset_train: Optional[int] = 1500
    data_subset_val: Optional[int] = 200
    data_subset_test: Optional[int] = 200

    # Split ratios used only when generating synthetic data (not CoNLL).
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Number of synthetic samples to generate when real data is unavailable.
    num_synthetic_samples: int = 1000

    # DataLoader worker processes.
    # Windows requires num_workers=0 (multiprocessing spawn issues with BERT).
    num_workers: int = 0


# ==============================================================================
# 5. PathConfig
# ==============================================================================

@dataclass
class PathConfig:
    """File system paths for checkpoints, logs, and outputs.

    All paths are derived from the project root so the project can be moved
    without breaking anything.
    """

    # Project root is the directory that contains this config.py file.
    project_root: str = os.path.dirname(os.path.abspath(__file__))

    # These are set in __post_init__ once project_root is known.
    checkpoint_dir: str = field(default="")
    best_model_path: str = field(default="")
    tensorboard_dir: str = field(default="")
    output_dir: str = field(default="")

    def __post_init__(self):
        """Derive concrete paths from the project root after dataclass init."""
        self.checkpoint_dir  = os.path.join(self.project_root, "checkpoints")
        self.best_model_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        self.tensorboard_dir = os.path.join(self.project_root, "runs")
        self.output_dir      = os.path.join(self.project_root, "output")

    def ensure_dirs(self):
        """Create all required directories if they do not already exist."""
        for dir_path in [self.checkpoint_dir, self.tensorboard_dir, self.output_dir]:
            os.makedirs(dir_path, exist_ok=True)


# ==============================================================================
# 6. NERConfig  (master config)
# ==============================================================================

@dataclass
class NERConfig:
    """Master configuration that bundles all sub-configs.

    Usage
    -----
        config = NERConfig()                  # all defaults
        config.training.num_epochs = 10       # override a single field
        config.paths.ensure_dirs()            # create output dirs
    """

    model:    ModelConfig    = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data:     DataConfig     = field(default_factory=DataConfig)
    paths:    PathConfig     = field(default_factory=PathConfig)

    # Label mappings exposed at the top level for convenience.
    label2id:        Dict[str, int] = field(default_factory=lambda: LABEL2ID.copy())
    id2label:        Dict[int, str] = field(default_factory=lambda: ID2LABEL.copy())
    num_labels:      int            = NUM_LABELS
    ignore_label_id: int            = IGNORE_LABEL_ID

    def __post_init__(self):
        """Keep derived fields in sync after any manual overrides."""
        # The model must know how many labels it needs to predict.
        self.model.num_labels = self.num_labels
        # The model and data pipeline must agree on sequence length.
        self.model.max_seq_length = self.data.max_seq_length
        # Create output directories so downstream code never has to worry.
        self.paths.ensure_dirs()
