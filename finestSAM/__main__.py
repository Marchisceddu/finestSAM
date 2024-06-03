import argparse
from config import cfg
from model import train, automatic_predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modello finestSAM, permette di effettuare un fine-tuning (--mode train) o di effettuare predizioni (--mode predict)") 
    parser.add_argument('--mode', choices=['train', 'predict'], required=True, help='Modalità di esecuzione: train o predict')
    args = parser.parse_args()

    switcher = {
        "train": train,
        "predict": lambda cfg: automatic_predictions(cfg, "../dataset/images/0.png")
    }
    switcher[args.mode](cfg)