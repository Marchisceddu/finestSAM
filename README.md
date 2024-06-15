# Shape_SAM

Questo progetto è stato realizzato come progetto di tesi per l'Università degli Studi di Cagliari da:

* [Marco Pilia](https://github.com/Marchisceddu)
* [Simone Dessi](https://github.com/Druimo)

Lo scopo principale è cercare di fare il fine tuning del modello Segment-Anything di MetaAI su un set di dati personalizzati in formato COCO.
Il codice sfrutta il framework Fabric di Lightning AI per fornire un implementazione efficiente del modello.

CONTINUARE LA DESCRIZIONE

## Creazione dataset:
* Aggiungere all'interno della cartella: [create_dataset/shp/](https://github.com/Marchisceddu/Progetto_Urbismap/blob/main/create_dataset/shp) i dati inseriti in cartelle formattate come segue:

    ```python
    shp/
    ├── image01/
    │   ├── image01.shp
    │   ├── ... # Resto dei file che servono per il file shp (.cpg, .dbf, .prj, .shx)
    │   ├── img.tif
    │   ├── img2.tif # Ci possono essere più tif georeferenziate per ogni file shp
    │   └── ...
    ├── image02/
    │   ├── image02.shp
    │   ├── ... # Resto dei file che servono per il file shp (.cpg, .dbf, .prj, .shx)
    │   ├── img.tif
    │   ├── img2.tif
    │   └── ...
    └── ...
    ```

* Eseguire il file [create_dataset/__main__.py](https://github.com/Marchisceddu/Progetto_Urbismap/blob/main/create_dataset/__main__.py):

    Args (opzionali):
    ```python
     --scegli_input (bool) default:False # Se True, permette di scegliere la cartella di input 
                                            # (deve essere formattata come la cartella shp)

     --mostra_output (bool) default:False # Se True, mostra l'output del dataset
     ```

    Run:

       python create_dataset --scegli_input False --mostra_output True

  questo creerà il dataset in formato COCO all'interno della cartella [dataset/](https://github.com/Marchisceddu/Progetto_Urbismap/tree/main/dataset/)

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

"train_type": str = "custom" or "11_iterations",
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
    "decay_factor": int,
    "steps": [int, int],
    "warmup_steps": int,
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
        "path": {
            "root_dir": str,
            "annotation_file": str,
        },
        "train": {
            "root_dir": str,
            "annotation_file": str,
        },
        "val": {
            "root_dir": str,
            "annotation_file": str,
        },

    "val_size": float (0-1), # Percentuale grandezza validation dataset
    "positive_points": int, # Numero punti positivi passati con __getitem__
    "negative_points": int, # Numero punti negativi passati con __getitem__
}
```

Predizioni:
```python
"approx_accuracy": float, # The approximation accuracy of the polygons
```

</details>

## Run model:

Eseguire il file [shape_SAM/__main__.py](https://github.com/Marchisceddu/Progetto_Urbismap/blob/main/shape_SAM/__main__.py)

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
    ```

### Predizioni manuali:

Per il momento non sono ancora disponibili, si possono effetuare sia tramite boxe sia tramite punti, c'è un piccolo esempio in prediction ma vanno aggiustate fornendo magai un interfaccia grafica per poter far disegnare i punti o le boxe sull'immagini da predirre

## Resources

- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [Lightning AI](https://github.com/Lightning-AI/lightning)
- [lightning-sam](https://github.com/luca-medeiros/lightning-sam)