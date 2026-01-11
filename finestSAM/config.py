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
    },
}

config_training = {
    "seed_dataloader": None,
    "batch_size": 1,
    "num_workers": 0,

    "num_epochs": 150,
    "eval_interval": 3,
    "prompts": {
        "use_boxes": False,
        "use_points": True,
        "use_masks": False,
    },
    "multimask_output": False,

    "opt": {
        "learning_rate": 1e-4,
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
            "decay_factor": 0.05, # lr * factor -> 8e-4 * 0.1 = 8e-5
            "epoch_patience": 3,
            "threshold": 1e-4,
            "cooldown": 0,
            "min_lr": 0,
            "warmup_steps": 250,
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
                    "mlp_lin1": False,
                    "mlp_lin2": False,
                },
            },
            "decoder": {
                "enabled": False,
                "lora_r": 4,
                "lora_alpha": 4,
                "lora_dropout": 0.1,
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
    "prompts": {
        "use_boxes": False,
        "use_points": True,
        "use_masks": False,
    },
    "multimask_output": False,
    "dataset": {
        "seed": 42,
        "use_cache": True,
        "sav": "sav.pth",
        "positive_points": 1,
        "negative_points": 0,
        "use_center": True,
        "snap_to_grid": True,
    }
}

config_inference = {
    "opacity": 0.9,
}

cfg_training = Box(config)
cfg_training.update(Box(config_training))

cfg_evaluation = Box(config)
cfg_evaluation.update(Box(config_evaluation))

cfg_inference = Box(config)
cfg_inference.update(Box(config_inference))