import argparse
from model import train, automatic_predictions

if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(description="Modello finestSAM, permette di effettuare un fine-tuning (--mode train) o di effettuare predizioni (--mode predict)")
    parser.add_argument('--mode', choices=['train', 'predict'], required=True, help='Modalità di esecuzione: train o predict')

    args, unknown = parser.parse_known_args()

    if args.mode == 'predict':
        from config import cfg_predict as cfg

        predict_parser = argparse.ArgumentParser()
        predict_parser.add_argument('--input', type=str, required=True, help='Path dell\'immagine da analizzare')
        predict_parser.add_argument('--approx_accuracy', type=float, default=cfg.approx_accuracy, help='Accuratezza approssimativa dei poligoni')
        predict_args = predict_parser.parse_args(unknown)
    elif args.mode == 'train':
        from config import cfg_train as cfg

        predict_args = None

    # Execute the mode selected
    switcher = {
        "train": train,
        "predict": lambda cfg: automatic_predictions(cfg, args.path, args.approx_accuracy)
    }
    switcher[args.mode](cfg)