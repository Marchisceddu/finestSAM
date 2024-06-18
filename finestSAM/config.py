from box import Box

config = {
    "device": "auto",
    "num_devices": "auto",
    "seed_device": 1337,
    "sav_dir": "sav",
    "out_dir": "out",

    "model": {
        "type": 'vit_b',
        "checkpoint": "sam_vit_b_01ec64.pth",
    },
}

config_train = {
    "seed_dataloader": 42,
    "batch_size": 1,
    "num_workers": 0,

    "train_type": "custom",
    "num_epochs": 100,
    "eval_interval": 10,
    "eval_improvement": 0.,
    "prompts": {
        "use_boxes": False,
        "use_points": True,
        "use_masks": False,
        "use_logits": False,
    },
    "multimask_output": True,

    "opt": {
        "learning_rate": 8e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": None,
        "warmup_steps": 250,
    },

    "losses": {
        "focal_ratio": 20.,
        "dice_ratio": 1.,
        "iou_ratio": 1.,
        "focal_alpha": 0.8,
        "focal_gamma": 2,
    },

    "model_layer": {
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
    },

    "dataset": {
        "auto_split": True,
        "path": {
            "root_dir": "../dataset/images",
            "annotation_file": "../dataset/annotations.json"
        },
        "train": {
            "root_dir": "../datasetCalcio",
            "annotation_file": "../datasetCalcio/annotations.json"
        },
        "val": {
            "root_dir": "../datasetC/val",
            "annotation_file": "../datasetC/val/annotations.json"
        },
        "val_size": 0.2,
        "positive_points": 1,
        "negative_points": 0,
    }
}

config_predict = {
    "approx_accuracy": 0.01,
}

cfg_train = Box(config)
cfg_train.update(Box(config_train))

cfg_predict = Box(config)
cfg_predict.update(Box(config_predict))