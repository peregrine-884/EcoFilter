import cv2
import os
import numpy as np
import random

def adjust_brightness(image, factor):
    """ 画像の明るさを調整する関数 """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def augment_images_in_folder(images_folder_path, labels_folder_path, num_augmentations=5):
    """ フォルダ内のすべての画像に対してランダムな明るさ変更を行う関数 """
    for filename in os.listdir(images_folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(images_folder_path, filename)
            annotation_filename = os.path.splitext(filename)[0] + ".txt"
            annotation_path = os.path.join(labels_folder_path, annotation_filename)

            if not os.path.exists(annotation_path):
                continue  # アノテーションファイルが存在しない場合はスキップ

            image = cv2.imread(image_path)

            for i in range(num_augmentations):
                factor = random.uniform(0.6, 1.4)  # 明るさ変更の係数をランダムに選択
                new_image = adjust_brightness(image, factor)
                new_image_filename = f"{os.path.splitext(filename)[0]}_brightness_{i}.jpg"
                new_image_path = os.path.join(images_folder_path, new_image_filename)
                new_annotation_path = os.path.join(labels_folder_path, os.path.splitext(new_image_filename)[0] + ".txt")

                cv2.imwrite(new_image_path, new_image)
                os.system(f"cp '{annotation_path}' '{new_annotation_path}'")

# 例：フォルダパスと水増しする枚数
images_folder_path = "data/train/images"
labels_folder_path = "data/train/labels"
num_augmentations = 3  # 1つの画像に対する水増しの枚数

# 処理実行
augment_images_in_folder(images_folder_path, labels_folder_path, num_augmentations)
