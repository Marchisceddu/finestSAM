from box import Box

config = {
    "device": "auto",
    "seed_device": 1337,
    "seed_dataloader": None,

    "num_devices": "auto",
    "batch_size": 2,
    "num_workers": 0,
    "sav_dir": "sav",
    "out_dir": "out",

    "train_type": "custom",
    "num_epochs": 200,
    "eval_interval": 2,
    "eval_improvement": 0.05,
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

    "losses": {
        "focal_ratio": 20.,
        "dice_ratio": 1.,
        "focal_alpha": 0.75,
        "focal_gamma": 2,
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
        "root_dir": "../dataset3/images",
        "annotation_file": "../dataset3/annotations.json",
        "val_size": 0,
        "positive_points": 4,
        "negative_points": 4,
    }
}

cfg = Box(config)
