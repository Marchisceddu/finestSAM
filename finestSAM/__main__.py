import argparse
from finestSAM.engine.trainer import call_train
from finestSAM.engine.predictor import call_predict
from finestSAM.engine.evaluator import call_test

if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(description="finestSAM model, allows fine-tuning (--mode train) or making predictions (--mode predict)")
    parser.add_argument('--mode', choices=['train', 'predict', 'test'], required=True, help='Execution mode: train, predict or test')
    args, unknown = parser.parse_known_args()

    if args.mode == 'train':
        from config import cfg_training as cfg

        train_parser = argparse.ArgumentParser()
        train_parser.add_argument('--dataset', type=str, required=True, help='Path of the dataset to use for training')
        train_args = train_parser.parse_args(unknown)
        
    elif args.mode == 'test':
        from config import cfg_evaluation as cfg

        test_parser = argparse.ArgumentParser()
        test_parser.add_argument('--dataset', type=str, required=True, help='Path of the dataset to use for testing')
        test_parser.add_argument('--checkpoint', type=str, default=None, help='Path to the checkpoint file (optional)')
        test_parser.add_argument('--model_type', type=str, default=None, choices=['vit_b', 'vit_l', 'vit_h'], help='Type of the model (vit_b, vit_l, vit_h) (optional)')
        test_args = test_parser.parse_args(unknown)

    elif args.mode == 'predict':
        from config import cfg_inference as cfg

        predict_parser = argparse.ArgumentParser()
        predict_parser.add_argument('--input', type=str, required=True, help='Path of the input image')
        predict_parser.add_argument('--opacity', type=float, default=None, help='Opacity of the mask')
        predict_parser.add_argument('--checkpoint', type=str, default=None, help='Path to the checkpoint file (optional)')
        predict_parser.add_argument('--model_type', type=str, default=None, choices=['vit_b', 'vit_l', 'vit_h'], help='Type of the model (vit_b, vit_l, vit_h) (optional)')
        predict_args = predict_parser.parse_args(unknown)

    # Execute the selected mode 
    switcher = {
        "train": lambda cfg: call_train(cfg, train_args.dataset),
        "test": lambda cfg: call_test(cfg, test_args.dataset, test_args.checkpoint, test_args.model_type),
        "predict": lambda cfg: call_predict(cfg, predict_args.input, predict_args.opacity, predict_args.checkpoint, predict_args.model_type)
    }
    switcher[args.mode](cfg)