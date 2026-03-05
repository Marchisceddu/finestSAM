import os
from box import Box

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

config = {
    "device": "auto",
    "num_devices": "auto",
    "num_nodes": 1,
    "precision": "16-mixed",
    "matmul_precision": "high",
    "seed_device": 1337,
    "sav_dir": os.path.join(BASE_DIR, "sav"),
    "out_dir": os.path.join(BASE_DIR, "out"),

    "model": {
        "type": 'vit_b',
        "checkpoint": "sam_vit_b_01ec64.pth",
        "img_size": 512,
        "compute_stats": True,
        "pixel_mean": None,
        "pixel_std": None, 
    },

    "model_layer": {
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": False,
            "mask_decoder": False,
        },
        "LORA": {
            "encoder": {
                "enabled": True,
                "lora_r": 4,
                "lora_alpha": 4,
                "lora_dropout": 0,
                "lora_bias": False,
                "lora_targets": {
                    "q_proj": True,
                    "k_proj": False,
                    "v_proj": True,
                    "out_proj": False,
                    "mlp_lin1": True,
                    "mlp_lin2": True,
                },
            },
            "decoder": {
                "enabled": False,
                "lora_r": 4,
                "lora_alpha": 4,
                "lora_dropout": 0,
                "lora_bias": False,
                "lora_targets": {
                    "q_proj": False,
                    "k_proj": False,
                    "v_proj": False,
                    "out_proj": False,
                    "mlp_lin1": False,
                    "mlp_lin2": False,
                    "hypernet_mlp": False,
                    "iou_head_mlp": False,
                },
            },
        },
    },
}

config_training = {
    "batch_size": 8,
    "num_workers": 0,

    "num_epochs": 30,
    "eval_interval": 1,
    "val_at_epoch_0": False,
    "print_images": 0,
    "prompts": {
        "use_boxes": True,
        "use_points": False,
        "use_masks": False,
    },
    "multimask_output": False,

    "opt": {
        "learning_rate": 4e-5,
        "weight_decay": 1e-4, 
    },

    "sched": {
        "type": "ReduceLROnPlateau",
        "LambdaLR": {
            "decay_factor": 10, # 1 / (cfg.sched.LambdaLR.decay_factor ** (mul_factor+1))
            "steps": None,
            "warmup_steps": 0,
        },
        "ReduceLROnPlateau": {
            "monitor": "val_loss", # "train_loss" or "val_loss"
            "decay_factor": 0.1, # lr * factor -> 8e-4 * 0.1 = 8e-5
            "epoch_patience": 3,
            "threshold": 1e-3,
            "cooldown": 3,
            "min_lr": 1e-7,
            "warmup_steps": 250,
        },
    },

    "losses": {
        "focal": {
            "enabled": False,
            "weight": 20.0,
            "gamma": 2.0,
        },
        "dice": {
            "enabled": True,
            "weight": 1.0,
        },
        "iou": {
            "enabled": True, 
            "weight": 1.0,
        },
        "cross_entropy": {
            "enabled": True, 
            "weight": 1.0,
        },
    },
    
    "metrics": {
        "iou": {"enabled": True},
        "dice": {"enabled": True},
        "hd95": {"enabled": True},
    },

    "dataset": {
        "seed": 42,
        "use_cache": True,
        "sav": "sav.pth",
        "val_size": 0.2,
        "positive_points": 1,
        "negative_points": 0,
        "use_center": True,
        "snap_to_grid": True,
    }
}

config_evaluation = {
    "batch_size": 1,
    "num_workers": 0,
    "print_images": "all",
    "prompts": {
        "use_boxes": True,
        "use_points": False,
        "use_masks": False,
    },
    "multimask_output": False,

    "losses": {
        "focal": {
            "enabled": False,
            "weight": 20.0,
            "gamma": 2.0,
        },
        "dice": {
            "enabled": True,
            "weight": 1.0,
        },
        "iou": {
            "enabled": True, 
            "weight": 1.0,
        },
        "cross_entropy": {
            "enabled": True, 
            "weight": 1.0,
        },
    },
    "metrics": {
        "iou": {"enabled": True},
        "dice": {"enabled": True},
        "hd95": {"enabled": True},
    },

    "dataset": {
        "seed": 42,
        "use_cache": True,
        "sav": "sav.pth",
        "positive_points": 1,
        "negative_points": 0,
        "use_center": True,
        "snap_to_grid": True,
    },
}

cfg_training = Box(config)
cfg_training.update(Box(config_training))

cfg_evaluation = Box(config)
cfg_evaluation.update(Box(config_evaluation))