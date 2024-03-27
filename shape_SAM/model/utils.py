import torch
import random
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def draw_image(image, masks, boxes, labels, alpha=0.4):
    image = torch.from_numpy(image).permute(2, 0, 1)
    if boxes is not None:
        image = draw_bounding_boxes(image, boxes, colors=['red'] * len(boxes), labels=labels, width=2)
    if masks is not None:
        colors = [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(len(masks))]
        image = draw_segmentation_masks(image, masks=masks, colors=colors, alpha=alpha)
    return image.numpy().transpose(1, 2, 0)