import argparse
from config import cfg
from model import train, automatic_predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utilizzo del modello shape_SAM, possibilità di fare il fine-tuning del modello o di fare delle predizioni automatiche") 
    parser.add_argument('--mode', choices=['train', 'predict'], required=True, help='Modalità di esecuzione: train o predict')
    args = parser.parse_args()

    switcher = {
        "train": train,
        "predict": lambda cfg: automatic_predictions(cfg, "../dataset/images/0.png")
    }
    switcher[args.mode](cfg)