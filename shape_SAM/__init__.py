import argparse
from .model.config import cfg
from .model.train import train
from .model.predictions import automatic_predictions
from ..create_dataset.dataset_functions import get_png_file_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utilizzo del modello shape_SAM, possibilità di fare il fine-tuning del modello o di fare delle predizioni automatiche") 
    parser.add_argument("--mode", action="store_true", help="train / predictions", default=None)
    args = parser.parse_args()

    if args.mode == "train":
        train(cfg)
    elif args.mode == "predictions":
        png_file_path = "../dataset/images/0.png"
        automatic_predictions(cfg, png_file_path)
    else:
        print("Errore: inserire un argomento valido (train / predictions)")
        exit(1)