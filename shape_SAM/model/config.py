from box import Box

config = {
    "seed_device": 1337,
    "seed_dataloader": None,

    "num_devices": "auto",
    "batch_size": 1,
    "num_workers": 2,
    "out_dir": "sav",

    "train_type": "custom",
    "num_epochs": 200,
    "eval_interval": 100,
    "custom_cfg": {
            "use_boxes": True,
            "use_points": True,
            "use_masks": True,
    },
    "cross-validation_cfg": { 
        "k_fold": 5,
    },

    "opt": {
        "learning_rate": 8e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 250,
    },

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
        "root_dir": "../dataset/images",
        "annotation_file": "../dataset/annotations.json",
        "val_size": 0.2,
        "positive_points": 1,
        "negative_points": 0,
    }
}

#ATTENZIONE
# train e val in dataset vanno eliminati in teoria e sostituiti con lo split del dataset in dataset.py

cfg = Box(config)
