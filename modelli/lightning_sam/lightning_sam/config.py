from box import Box

config = {
    "num_devices": 4,
    "batch_size": 12,
    "num_workers": 4,
    "num_epochs": 20,
    "k_fold": 5,
    "eval_interval": 2,
    "out_dir": "out/training",
    "opt": {
        "learning_rate": 8e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 250,
    },
    "model": {
        "type": 'vit_h',
        "checkpoint": "sav/sam_vit_h_4b8939.pth",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
    },
    "dataset": {
        "root_dir": "./dataset/coco/images",
        "annotation_file": "./dataset/coco/annotations.json",
        
        "train": {
            "root_dir": "./dataset/train",
            "annotation_file": "./dataset/train.json"
        },
        "val": {
            "root_dir": "./dataset/val",
            "annotation_file": "./dataset/val.json"
        }
    }
}

cfg = Box(config)
