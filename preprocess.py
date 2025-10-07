import cv2
import numpy as np
import os

class image_preprocessor:
    def __init__(self, image_path=None, binarize=False, size=(28, 28)):
        self.size = size
        self.binarize = binarize
        self.image_path = image_path

    def preprocess(self):
        if not self.image_path or not os.path.exists(self.image_path):
            raise ValueError(f"Image at path {self.image_path} could not be read.")

        img = cv2.imread(self.image_path)
        if img is None:
            raise ValueError(f"Image at path {self.image_path} could not be read.")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        if self.binarize:
            gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        # Return full-size preprocessed gray image (not resized). Segmentation will crop.
        return gray

    def show_image(self, image):
        cv2.imshow('Preprocessed Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()