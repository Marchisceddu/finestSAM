from box import Box

config = {
    "device": "auto",
    "num_devices": "auto",
    "num_nodes": 1, # Numero di nodi GPU per l'addestramento distribuito
    "seed_device": 1337,
    "sav_dir": "sav",
    "out_dir": "out",

    "model": {
        "type": 'vit_b',
        "checkpoint": "sam_vit_b_01ec64.pth",
    },
}

config_train = {
    "seed_dataloader": None,
    "batch_size": 1,
    "num_workers": 0,

    "num_epochs": 500,
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
    },

    "sched": {
        "type": "ReduceLROnPlateau", #"ReduceLROnPlateau" LambdaLR
        "LambdaLR": {
            "decay_factor": 10, # 1 / (cfg.sched.LambdaLR.decay_factor ** (mul_factor+1))
            "steps": None, # il primo step dev'essere maggiore di warmup_steps, stemps = list[int]
            "warmup_steps": 0,
        },
        "ReduceLROnPlateau": {
            "decay_factor": 0.05, #lr * factor -> 8e-4 * 0.1 = 8e-5
            "epoch_patience": 15,
            "threshold": 1e-4,
            "cooldown": 0,
            "min_lr": 2.0000000000000003e-06, # fa fare massimo 2 abbassate di lr
        },
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
        "seed": 42,
        "split_path": {
            "root_dir": "../dataset",
            "images_dir": "images",
            "annotation_file": "annotations.json",
            "sav": "sav.pth",
            "val_size": 0.2,
        },
        "no_split_path": {
            "train": {
                "root_dir": "../dataset/train",
                "images_dir": "images",
                "annotation_file": "annotations.json",
                "sav": "sav.pth"
            },
            "val": {
                "root_dir": "../dataset/val",
                "images_dir": "images",
                "annotation_file": "annotations.json",
                "sav": "sav.pth"
            },
        },
        "positive_points": 4,
        "negative_points": 4,
        "use_center": True, # il primo punto positivo sarà sempre il centro di massa
    }
}

config_predict = {
    "approx_accuracy": 0.01,
    # inserire la trasparenza delle maschere da disegnare sopra
}

cfg_train = Box(config)
cfg_train.update(Box(config_train))

cfg_predict = Box(config)
cfg_predict.update(Box(config_predict))