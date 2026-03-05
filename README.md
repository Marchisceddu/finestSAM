# finestSAM

**finestSAM** is a configuration-driven framework for fine-tuning Meta AI's Segment Anything Model (SAM) on custom datasets in COCO format.

Developed by [`Marco Pilia`](https://github.com/Marchisceddu) and [`Simone Dessi`](https://github.com/Druimo).

The goal is to make SAM fine-tuning reproducible and highly controllable, letting you decide (via [`finestSAM/config.py`](https://github.com/WholeNow/finestSAM/blob/main/finestSAM/config.py)) small but important details such as:

- Training strategy: classic fine-tuning or LoRA adapters (encoder and/or decoder), with selectable target submodules.
- What prompts drive supervision: boxes / points / masks (train and evaluation can differ).
- Which SAM components are frozen (image encoder / prompt encoder / mask decoder).
- Optimization knobs: losses and metrics toggles, schedulers, precision, and device selection (including `auto`).
- Data handling: dataset split/caching options and optional dataset-driven normalization statistics.

Built on **Lightning AI Fabric**, it is designed to scale cleanly from local runs to multi-device training while keeping compatibility with SAM inference and automatic prediction workflows.

## Dataset
You can structure your dataset in two ways, depending on whether you want the script to automatically split it into training and validation sets or if you prefer to provide them manually.

### Option 1: Auto-Split
If you want the script to handle the split, organize your folder as follows:

```
dataset/
└── data/
    ├── images/           # Folder containing all images
    │   ├── 0.png
    │   └── ...
    └── annotations.json  # COCO annotations for all images
```

### Option 2: Pre-Split
If you already have separate training and validation sets:

```
dataset/
├── train/
│   ├── images/
│   └── annotations.json
├── val/
│   ├── images/
│   └── annotations.json
└── test/
    ├── images/
    └── annotations.json
```

## Setup

Here are the steps to follow:

1. **Download the SAM model checkpoint**  
   The instructions for downloading the SAM model checkpoint can be found in the [`finestSAM/sav/`](https://github.com/WholeNow/finestSAM/blob/main/finestSAM/sav/) directory.

2. **Install necessary dependencies:**
    To install the required dependencies, run the following command:
    ```bash
    pip install -r requirements.txt
    ```

## Config

The hyperparameters required for the model are specified in [`finestSAM/config.py`](https://github.com/WholeNow/finestSAM/blob/main/finestSAM/config.py).

<details>
<summary> <b>Configuration Overview</b> </summary>

### **General**
- `device`: (String) The hardware to run on (`"cpu"`, `"cuda"`, `"mps"`, `"gpu"`, `"tpu"`, `"auto"`).
- `num_devices`: (String/Int/List) Number of devices to train on (int), which GPUs to train on (list or str), or `"auto"`. The value applies per node.
- `num_nodes`: (Int) Number of GPU nodes for distributed training.
- `precision`: (String/Int) Controls the floating-point precision used during model training and inference.
    - Accepted values: `64` (double), `32` (full), `16-mixed` (half precision AMP), `bf16-mixed` (bfloat16 AMP).  
      *Other supported values:* `16`, `bf16`, `transformer-engine`, `transformer-engine-float16`, `16-true`, `bf16-true`, `32-true`, `64-true`.
    - If `None`, defaults will be used based on the device.
- `matmul_precision`: (String) Matrix multiplication precision for Tensor Cores (`"medium"`, `"high"`, `"highest"`).
- `seed_device`: (Int) Seed for device reproducibility (or None).
- `sav_dir`: (String) Output folder for model saves.
- `out_dir`: (String) Output folder for predictions.
- `model`:
    - `type`: (String) Model type (`"vit_h"`, `"vit_l"`, `"vit_b"`).
    - `checkpoint`: (String) Path to the .pth checkpoint file.
    - `img_size`: (Int) Input image size for the model (default `512`).
    - `compute_stats`: (Bool) If `True`, automatically compute `pixel_mean` and `pixel_std` from the training dataset.
    - `pixel_mean`: (List[Float]) Mean values for image normalization. If `None`, defaults to ImageNet mean.
    - `pixel_std`: (List[Float]) Standard deviation values for image normalization. If `None`, defaults to ImageNet std.

### **Training** / **Evaluation**
- `batch_size`: (Int) Batch size for images.
- `num_workers`: (Int) Number of subprocesses for data loading.
- `num_epochs`: (Int) Number of training epochs.
- `eval_interval`: (Int) Interval (in epochs) for validation.
- `val_at_epoch_0`: (Bool) If `True`, runs a validation pass once before training starts (epoch 0). Only applies when `eval_interval > 0`.
- `print_images`: (Int/String) Controls saving qualitative validation outputs.
    - `0`: disables saving.
    - Positive integer `N`: saves up to `N` validation batches (uniformly sampled) under `out_dir/images/`.
    - `"all"`: saves every validation batch under `out_dir/images/`.
- `prompts`:
    - `use_boxes`: (Bool) Use bounding boxes for training.
    - Notes: The box is randomly expanded/contracted by jittering its coordinates; the jitter scale is computed as 10% of the box side length and capped at a maximum of 20 pixels.
    - `use_points`: (Bool) Use points for training.
    - `use_masks`: (Bool) Use mask annotations for training.

- `multimask_output`: (Bool) Enable multimask output.
- `opt`:
    - `learning_rate`: (Float) Learning rate.
    - `weight_decay`: (Float) Weight decay.
- `sched`:
    - `type`: (String) Scheduler type (`"ReduceLROnPlateau"` or `"LambdaLR"`).
    - `LambdaLR`:
        - `decay_factor`: (Float) Learning rate decay factor.
        - `steps`: (List[Int]) List of steps for decay.
        - `warmup_steps`: (Int) Number of warmup epochs.
    - `ReduceLROnPlateau`:
        - `monitor`: (String) Which signal drives LR scheduling (`"train_loss"` or `"val_loss"`). When set to `"val_loss"`, the scheduler is stepped only on validation epochs.
        - `decay_factor`: (Float) Learning rate decay factor.
        - `epoch_patience`: (Int) Patience for LR decay.
        - `threshold`: (Float) Threshold for measuring the new optimum.
        - `cooldown`: (Int) Number of epochs to wait before resuming normal operation.
        - `min_lr`: (Float) Minimum learning rate.
        - `warmup_steps`: (Int) Number of warmup epochs.
- `losses`:
    - `focal`:
        - `enabled`: (Bool) Enable focal loss.
        - `weight`: (Float) Weight of focal loss.
        - `gamma`: (Float) Gamma value for focal loss.
    - `dice`:
        - `enabled`: (Bool) Enable dice loss.
        - `weight`: (Float) Weight of dice loss.
    - `iou`:
        - `enabled`: (Bool) Enable IoU loss.
        - `weight`: (Float) Weight of IoU loss.
    - `cross_entropy`:
        - `enabled`: (Bool) Enable Cross Entropy loss.
        - `weight`: (Float) Weight of Cross Entropy loss.
- `metrics`:
    - `iou`:
        - `enabled`: (Bool) Enable IoU metric.
    - `dice`:
        - `enabled`: (Bool) Enable Dice Score metric.
    - `hd95`:
        - `enabled`: (Bool) Enable 95th percentile Hausdorff distance metric (HD95).
- `model_layer`:
    - `freeze`:
        - `image_encoder`: (Bool) Freeze image encoder.
        - `prompt_encoder`: (Bool) Freeze prompt encoder.
        - `mask_decoder`: (Bool) Freeze mask decoder.
    - `LORA`:
        - `encoder`:
            - `enabled`: (Bool) Enable LoRA for the image encoder.
            - `lora_r`: (Int) Rank of the LoRA matrices (0 to disable).
            - `lora_alpha`: (Float) Scaling factor for LoRA weights (acts like a specific learning rate for adapters).
            - `lora_dropout`: (Float) Dropout applied to LoRA input.
            - `lora_bias`: (Bool) Enable bias in LoRA layers.
            - `lora_targets`:
                - `q_proj`: (Bool) Apply to Query projection in attention.
                - `k_proj`: (Bool) Apply to Key projection in attention.
                - `v_proj`: (Bool) Apply to Value projection in attention.
                - `out_proj`: (Bool) Apply to Output projection in attention.
                - `mlp_lin1`: (Bool) Apply to the first linear layer of the MLP.
                - `mlp_lin2`: (Bool) Apply to the second linear layer of the MLP.
        - `decoder`:
            - `enabled`: (Bool) Enable LoRA for the mask decoder.
            - `lora_r`: (Int) Rank of the LoRA matrices (0 to disable).
            - `lora_alpha`: (Float) Scaling factor for LoRA weights (acts like a specific learning rate for adapters).
            - `lora_dropout`: (Float) Dropout applied to LoRA input.
            - `lora_bias`: (Bool) Enable bias in LoRA layers.
            - `lora_targets`:
                - `q_proj`: (Bool) Apply to Query projection in attention.
                - `k_proj`: (Bool) Apply to Key projection in attention.
                - `v_proj`: (Bool) Apply to Value projection in attention.
                - `out_proj`: (Bool) Apply to Output projection in attention.
                - `mlp_lin1`: (Bool) Apply to first linear layer of MLPs.
                - `mlp_lin2`: (Bool) Apply to second linear layer of MLPs.
                - `hypernet_mlp`: (Bool) Apply to hypernetworks MLPs.
                - `iou_head_mlp`: (Bool) Apply to IoU prediction head.

### **Dataset**
- `seed`: Seed for dataset operations.
- `use_cache`: (Bool) Use cached dataset metadata.
- `sav`: Filename for saving dataset cache.
- `val_size`: (Float) Validation split percentage.
- `positive_points`: Number of positive points per mask.
- `negative_points`: Number of negative points per mask.
- `use_center`: (Bool) Use the mask center as a positive key point (most significant point).
- `snap_to_grid`: (Bool) Align the center point to the prediction grid used by the automatic predictor.

### **Prediction**
- `opacity`: Transparency of predicted masks (0.0 - 1.0).

</details>

## Run model

To execute the file [`finestSAM/__main__.py`](https://github.com/WholeNow/finestSAM/blob/main/finestSAM/__main__.py), use the following command-line arguments.

> [!TIP]
> Check out the provided notebooks for easy experimentation:
> - [`train.ipynb`](notebooks/train.ipynb) for training
> - [`test.ipynb`](notebooks/test.ipynb) for testing

### **Training the Model:**
Run the training process by specifying the mode and the dataset path:

```bash
python -m finestSAM --mode "train" --dataset "path/to/dataset"
```

### **Testing:**
To evaluate the model on a test dataset, use the `test` mode. You can optionally specify a checkpoint, the model type, and how many qualitative samples to save:

```bash
python -m finestSAM --mode "test" --dataset "path/to/test_dataset"
```

With specific checkpoint and model type:
```bash
python -m finestSAM --mode "test" --dataset "path/to/test_dataset" --checkpoint "path/to/checkpoint.pth" --model_type "vit_b" --output_images "all"
```

## To-Do List

- [ ] Adding Gradient Accumulation support.
- [ ] Add support for more SAM variants.

## Resources

- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [Lightning AI](https://github.com/Lightning-AI/lightning)
- [lightning-sam](https://github.com/luca-medeiros/lightning-sam)

## License
The model is licensed under the [Apache 2.0 license](https://github.com/WholeNow/finestSAM/blob/main/LICENSE.txt).