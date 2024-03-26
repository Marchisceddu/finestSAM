# Creazione dataset:
* Aggiungere all'interno della cartella: [create_dataset/shp/](https://github.com/Marchisceddu/Progetto_Urbismap/blob/main/create_dataset/shp) i dati inseriti in cartelle formattate come segue:

    ```python
    shp/
    ├── image01/
    │   ├── image01.shp
    │   ├── ... # resto dei file che servono per il file shp (.cpg, .dbf, .prj, .shx)
    │   ├── img.tif
    │   ├── img2.tif     # Ci possono essere più tif georeferenziate per ogni file shp
    │   └── ...
    ├── image02/
    │   ├── image02.shp
    │   ├── ... # resto dei file che servono per il file shp (.cpg, .dbf, .prj, .shx)
    │   ├── img.tif
    │   ├── img2.tif
    │   └── ...
    └── ...
    ```

* Eseguire il file [create_dataset/__init__.py](https://github.com/Marchisceddu/Progetto_Urbismap/blob/main/create_dataset/__init__.py):

    Args (opzionali):
    ```python
        --scegli_input (bool) default:False # Se True, permette di scegliere la cartella di input 
                                            # (deve essere formattata come la cartella shp)

        --mostra_output (bool) default:False # Se True, mostra l'output del dataset
     ```

    Run:

       python -m create_dataset --scegli_input False --mostra_output True

  questo creerà il dataset in formato COCO all'interno della cartella [dataset/](https://github.com/Marchisceddu/Progetto_Urbismap/tree/main/dataset/)

# Setup:

* E' necessario scaricare il checkpoint del modello SAM, la spiegazione è presente in [shape_SAM/sav/](https://github.com/Marchisceddu/Progetto_Urbismap/blob/main/shape_SAM/sav/)

* E' presente un file di configurazione -> [shape_SAM/model/config.py](https://github.com/Marchisceddu/Progetto_Urbismap/blob/main/shape_SAM/model/config.py) (permette di poter modificare diversi aspetti del modello e del train)

# Train:

*  Eseguire il file [shape_SAM/train.py](https://github.com/Marchisceddu/Progetto_Urbismap/blob/main/shape_SAM/train.py):

    Run:

        python shape_SAM/train.py