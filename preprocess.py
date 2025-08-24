import cv2
import numpy as np

class image_preprocessor:
    def __init__(self,image_path=None, binarize=False, size=(28, 28)):
        self.size = size
        self.binarize = binarize
        self.image_path = image_path
        # self.image = self.preprocess()

    def preprocess(self):
        # Read the image
        img = cv2.imread(self.image_path)
        if img is None:
            raise ValueError(f"Image at path {self.image_path} could not be read.")
        
        # Grayscale Image conversion:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize the image:
        # img = cv2.resize(img, self.size)

        if self.binarize:
            # Binarize the image
            pass
        
        print(f"Image preprocessed: {self.size}, Binarize: {self.binarize}")
        return img
    
    def show_image(self, image):
        cv2.imshow('Preprocessed Image', image)
        cv2.waitKey(0)