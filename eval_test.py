import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import NERConfig, ModelConfig, TrainingConfig, DataConfig, PathConfig
from src.data.ner_dataset import create_dataloaders
from src.model.ner_model import BroadcastNERModel
from src.training.trainer import NERTrainer
from transformers import BertTokenizerFast

def main():
    model_cfg = ModelConfig(model_name="bert-base-cased")
    train_cfg = TrainingConfig(batch_size=16)
    data_cfg = DataConfig()
    path_cfg = PathConfig()
    
    tokenizer = BertTokenizerFast.from_pretrained(
        model_cfg.model_name, 
        do_lower_case=False
    )
    
    print("Preparing test dataset...")
    _, _, test_loader = create_dataloaders(
        tokenizer=tokenizer,
        config=data_cfg,
        batch_size=train_cfg.batch_size,
        num_workers=0,
    )
    
    print(f"\nLoading best model from {path_cfg.best_model_path}...")
    model = BroadcastNERModel.load_model(path_cfg.best_model_path, device="cpu")
    
    trainer = NERTrainer(
        model=model,
        train_loader=test_loader,
        val_loader=None,
        config=NERConfig(),
        device="cpu"
    )
    
    eval_results = trainer.evaluate(test_loader)
    print("\n  TEST SET EVALUATION COMPLETE")

if __name__ == "__main__":
    main()
