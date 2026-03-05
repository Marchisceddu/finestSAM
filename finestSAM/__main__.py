import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finestSAM.run.trainer import call_train
from finestSAM.run.evaluator import call_test

if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(description="finestSAM model, allows fine-tuning (--mode train) or evaluation (--mode test) of the model.")
    parser.add_argument('--mode', choices=['train', 'test'], required=True, help='Execution mode: train or test')
    args, unknown = parser.parse_known_args()

    if args.mode == 'train':
        from finestSAM.config import cfg_training as cfg

        train_parser = argparse.ArgumentParser()
        train_parser.add_argument('--dataset', type=str, required=True, help='Path of the dataset to use for training')
        train_args = train_parser.parse_args(unknown)
        
    elif args.mode == 'test':
        from finestSAM.config import cfg_evaluation as cfg

        test_parser = argparse.ArgumentParser()
        test_parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset to use for testing')
        test_parser.add_argument('--checkpoint', type=str, default=None, help='Path to the checkpoint file (optional)')
        test_parser.add_argument('--model_type', type=str, default=None, choices=['vit_b', 'vit_l', 'vit_h'], help='Type of the model (vit_b, vit_l, vit_h) (optional)')
        test_parser.add_argument('--output_images', type=str, default=None, help='Number of qualitative samples to save in the out folder. Can be "all", 0, or an integer.')
        test_args = test_parser.parse_args(unknown)

    # Execute the selected mode 
    switcher = {
        "train": lambda cfg: call_train(cfg, train_args.dataset),
        "test": lambda cfg: call_test(cfg, test_args.dataset, test_args.checkpoint, test_args.model_type, test_args.output_images),
    }
    switcher[args.mode](cfg)