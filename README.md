# Creazione dataset:
* Aggiungere all'interno della cartella: [create_dataset/shp/*]() delle cartelle formattate come segue:

shp/

├── image01/

│   ├── image01.shp

│   ├── #resto dei file che servono per il file shp

│   ├── 0.tif

│   ├── 1.tif     #Ci possono essere più tif georeferenziate per ogni file shp

│   └── ...

├── image02/

│   ├── image02.shp

│   ├── #resto dei file che servono per il file shp

│   ├── 0.tif

│   ├── 1.tif

│   └── ...

└── ...

* Eseguire il file [create_dataset/__init__.py]():

questo creerà il dataset in formato COCO all'interno della cartella [dataset/coco/]()
