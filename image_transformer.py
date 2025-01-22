import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


def to_fourier(dataset_path, new_dir=r'fourier\train'):

    def transformer(image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Load the image as gray

        if image is None:
            raise FileNotFoundError(f"Image at path '{image_path}' not found.")

        dft = np.fft.fft2(image)  # 2D Fourier Transform
        dft_shift = np.fft.fftshift(dft)  # Shift the zero frequency component to the center
        magnitude = np.abs(dft_shift)

        return magnitude

    

    images = os.listdir(dataset_path)
    dataset_path = Path(dataset_path)  # Replace with your dataset path
    parent_dir = dataset_path.parent  # Go one level up
    parent_dir = parent_dir.parent
    new_dataset_path = os.path.join(parent_dir, new_dir)
    os.makedirs(new_dataset_path)
    for image in images:
        transformed = transformer(os.path.join(dataset_path, image))
        out_path = os.path.join(parent_dir, new_dir, image[:-4]+'.npy').replace('.', '_', 1)
        print(out_path)
        np.save(out_path, transformed)



#to_fourier
dataset_path = r'D:\TG\RGB\train'
to_fourier(dataset_path)