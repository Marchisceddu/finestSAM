import os
import matplotlib.pyplot as plt
from torchvision.datasets import CocoDetection
from torchvision import transforms
from PIL import ImageDraw


def display_COCO(images_path, annotation_path):
    # Transforming images to tensors
    transform = transforms.ToTensor()

    # Upload the dataset from the path and the annotation file
    coco_dataset = CocoDetection(root = images_path, annFile = annotation_path, transform = transform)

    num_images_to_display = len(os.listdir(images_path))
    for i in range(num_images_to_display):
        # Upload the image and the targets from the dataset 
        image, targets = coco_dataset[i]

        # Convert the image to PIL format
        image_pil = transforms.ToPILImage()(image)

        # Prepare the image for drawing
        draw = ImageDraw.Draw(image_pil)

        # Adding the labels to the image
        for target in targets:
            # The COCO annotations are in the format of (x, y, width, height)
            segmentation = target['segmentation']
            for seg in segmentation:
                # Convert the segmentation to a list of points
                points = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
                # Draw the bounding box
                draw.line(points, fill='red', width=2)

        # Visualize the image
        plt.figure()
        plt.imshow(image_pil)
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    # Example of use
    display_COCO("../../dataset/images/", "../../dataset/annotations.json") # the path must start from the folder where the .py file is located