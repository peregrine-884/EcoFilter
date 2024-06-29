import cv2
import os
import numpy as np
import random
import shutil

def adjust_brightness(image, factor):
    """ Function to adjust the brightness of an image. """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def augment_images_in_folder(images_folder_path, labels_folder_path, num_augmentations=5):
    """ Function to randomly adjust the brightness of all images in a folder. """
    for filename in os.listdir(images_folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(images_folder_path, filename)
            base_filename, file_extension = os.path.splitext(filename)
            annotation_path = os.path.join(labels_folder_path, f"{base_filename}.txt")

            if not os.path.exists(annotation_path):
                continue  # Skip if annotation file does not exist

            image = cv2.imread(image_path)

            for i in range(num_augmentations):
                factor = random.uniform(0.6, 1.4)  # Randomly select brightness adjustment factor
                new_image = adjust_brightness(image, factor)
                new_image_filename = f"{base_filename}_brightness_{i}{file_extension}"
                new_image_path = os.path.join(images_folder_path, new_image_filename)
                new_annotation_filename = f"{base_filename}_brightness_{i}.txt"
                new_annotation_path = os.path.join(labels_folder_path, new_annotation_filename)

                cv2.imwrite(new_image_path, new_image)
                shutil.copy(annotation_path, new_annotation_path)

# Example: Folder paths and the number of augmentations to generate
images_folder_path = "data/valid/images"
labels_folder_path = "data/valid/labels"
num_augmentations = 3  # Number of augmentations per image

# Run the augmentation process
augment_images_in_folder(images_folder_path, labels_folder_path, num_augmentations)
