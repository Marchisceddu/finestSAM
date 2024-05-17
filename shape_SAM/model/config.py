from box import Box

config = {
    "seed_device": 1337,
    "seed_dataloader": None,
    "num_devices": "auto",
    "batch_size": 1,
    "num_workers": 2,
    "num_epochs": 200,
    "k_fold": 5,
    "eval_interval": 100,
    "out_dir": "sav",
    "opt": {
        "learning_rate": 8e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 250,
    },
    "use_boxes": True,
    "use_points": False,
    "use_masks": True,
    "model": {
        "type": 'vit_h',
        "checkpoint": "sam_vit_h_4b8939.pth",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
    },
    "dataset": {
        "root_dir": "../../dataset/images",
        "annotation_file": "../../dataset/annotations.json",
        "positive_points": 1,
        "negative_points": 0,
        
        "train": {
            "root_dir": "../../dataset/train",
            "annotation_file": "../../dataset/train.json"
        },
        "val": {
            "root_dir": "../../dataset/val",
            "annotation_file": "../../dataset/val.json"
        }
    }
}

#ATTENZIONE
# train e val in dataset vanno eliminati in teoria e sostituiti con lo split del dataset in dataset.py
# Ho messo 1 e zero perchè nel paper c'è scritto che usa solo un punto ma non ho capito nulla

cfg = Box(config)
