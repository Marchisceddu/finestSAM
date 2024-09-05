# finestSAM

Questo progetto è stato realizzato come progetto di tesi per l'Università degli Studi di Cagliari da:

* [Marco Pilia](https://github.com/Marchisceddu)
* [Simone Dessi](https://github.com/Druimo)

Lo scopo principale è cercare di fare il fine tuning del modello Segment-Anything di MetaAI su un set di dati personalizzati in formato COCO, riuscendo a fornire un'implementazione efficacie per quanto riguarda le predizioni tramite il predittore automatico di SAM.
Il codice sfrutta il framework Fabric di Lightning AI per fornire un implementazione efficiente del modello.

## Importazione Dataset:

Inserire un dataset formato coco all'interno della main dir, la cartella dev'essere formattata come segue:

    ```python
    datset/
    ├── images/ # Le immagini del dataset
    │   ├── 0.png
    │   ├── 1.png
    │   └── ...
    └── annotations.json # annotazioni formato coco
    ```

 Per specificare al modello il nome del dataset inserito modificare le voci correlate come spiegato nella sezione Config (mettere link), di default viene cercata la cartella "dataset".

## Setup:

1. Scaricare il checkpoint del modello SAM, la spiegazione è presente in [shape_SAM/sav/](https://github.com/Marchisceddu/Progetto_Urbismap/blob/main/shape_SAM/sav/)

2. Installare le dipendenze necessarie:

    * Installare tramite pip il file [requirements.txt](https://github.com/Marchisceddu/Progetto_Urbismap/requirements.txt)

          pip install -r requirements.txt

    * Creare un ambiente conda tramite il file [environment.yaml](https://github.com/Marchisceddu/Progetto_Urbismap/environment.yaml)

          conda env create -f environment.yml

## Config:

Per cambiare le impostazioni modificare il file [Shape_SAM/config.py](https://github.com/Marchisceddu/Progetto_Urbismap/blob/main/hape_SAM/config.py)

<details>

<summary> Struttura: </summary>
<br>

Generali:
```python
"device": str = "auto" or "gpu" or "cpu", # Hardware su cui eseguire il modello (non è supportata mps, se si usa un mac m1 impostare su cpu)
"num_devices": int # Numero di dispositivi da utilizzare
            or (list str) # definire queli GPU utilizzare
            or str = "auto",
"num_nodes": int, # Numero di nodi GPU per l'addestramento distribuito
"seed_device": int / None per random,
"sav_dir": str, # Cartella di output per i salvataggi
"out_dir": str, # Cartella di output per le predizioni

"model": {
    "type": str = "vit_h" or "vit_l" or "vit_b",
    "checkpoint": str, # Nome checkpoint, formato -> nome.pth
},
```

Train:
```python
"seed_dataloader": int / None per random,
"batch_size": int, # Grandezza batch delle immagini
"num_workers": int, # Quanti sottoprocessi utilizzare per il caricamento dei dati (0 -> i dati verranno caricati nel processo principale)

"num_epochs": int, # Numero di epoche di train
"eval_interval": int, # Intervallo di validazione
"eval_improvement": float (0-1), # Percentuale oltre il quale avviene il salvataggio
"prompts": {
    "use_boxes": bool, # Se True usa le boxe per il train
    "use_points": bool, # Se True usa i punti per il train
    "use_masks": bool, # Se True usa le annotazioni per il train
    "use_logits": bool, # Se True usa i logits dell'epoca precedente (se True viene ignorato use_masks)
},
"multimask_output": bool,

"opt": {
    "learning_rate": int,
    "weight_decay": int,
},

"sched": {
        "type": str = "ReduceLROnPlateau" or  "LambdaLR"
        "LambdaLR": {
            "decay_factor": int, # fattore di dacadimento del lr funziona tramite la formula -> 1 / (decay_factor ** (mul_factor+1))
            "steps": list int, # lista che indica ogni quante epoche deve decadere il lr (il primo step dev'essere maggiore di warmup_steps)
            "warmup_steps": int, # aumenta il lr fino ad arrivare a stabilizzarlo in questo numero d'epoche
        },
        "ReduceLROnPlateau": {
            "decay_factor": float (0-1), # fattore di dacadimento del lr funziona tramite la formula -> lr * factor -> 8e-4 * 0.1 = 8e-5
            "epoch_patience": int, # Pazienza per il decadimento del lr
            "threshold": float,
            "cooldown": int,
            "min_lr": int,
        },
    },

"losses": {
    "focal_ratio": float, # Peso di Focal loss sulla loss totale
    "dice_ratio": float, # Peso di Dice loss sulla loss totale
    "iou_ratio": float, # Peso di Space IoU loss sulla loss totale
    "focal_alpha": float, # Valore di alpha per la Focal loss
    "focal_gamma": int, # Valore di gamma per la Focal loss
},

"model_layer": {
    "freeze": {
        "image_encoder": bool, # Se True freez del livello
        "prompt_encoder": bool, # Se True freez del livello
        "mask_decoder": bool, # Se True freez del livello
    },
},

"dataset": {
    "auto_split": bool, # Se True verra usato il dataset presente in path ed effettuare uno split per la validation della dimensione di val_size 
    "seed": 42,
    "split_path": {
        "root_dir": str,
        "images_dir": str,
        "annotation_file": str,
        "sav": str, # Eliminare il sav vecchio ad ogni cambio di impostazione
        "val_size": float (0-1), # Percentuale grandezza validation dataset
    },
    "no_split_path": {
        "train": {
            "root_dir": str,
            "images_dir": str,
            "annotation_file": str,
            "sav": str, # Eliminare il sav vecchio ad ogni cambio di impostazione
        },
        "val": {
            "root_dir": str,
            "images_dir": str,
            "annotation_file": str,
            "sav": str, # Eliminare il sav vecchio ad ogni cambio di impostazione
        },
    },
    "positive_points": int, # Numero punti positivi passati con __getitem__
    "negative_points": int, # Numero punti negativi passati con __getitem__
    "use_center": True, # il primo punto positivo sarà sempre il centro di massa
    "snap_to_grid": True, # allinea il centro di massa alla griglia di predizione utilizzata dal presdittore automatico
}
```

Predizioni:
```python
"approx_accuracy": float, # The approximation accuracy of the polygons
"opacity": float, 
```

</details>

## Run model:

Eseguire il file [finestSAM/__main__.py](https://github.com/Marchisceddu/Progetto_Urbismap/blob/main/finestSAM/__main__.py)

Args (obbligatori):

```python
--mode (str)
```

### Train

```python
--mode "train"
```

### Predizioni automatiche:

```python
--mode "predict" --input "percorso/image.png"
```

* Args (opzionali - modificabili anche in config):

    ```python
    --approx_accuracy (float) default:0.01 # The approximation accuracy of the polygons
    --opacity (float) default:0.9 
    ```

## Resources

- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [Lightning AI](https://github.com/Lightning-AI/lightning)
- [lightning-sam](https://github.com/luca-medeiros/lightning-sam)