import cv2
import numpy as np

class image_preprocessor:
    def __init__(self, image_path=None, binarize=False, size=(28, 28)):
        self.size = size
        self.binarize = binarize
        self.image_path = image_path

    def preprocess(self):
        img = cv2.imread(self.image_path)
        if img is None:
            raise ValueError(f"Image at path {self.image_path} could not be read.")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # For noise reduction
        img = cv2.GaussianBlur(img, (3, 3), 0)

        if self.binarize:
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        print(f"Image preprocessed: {self.size}, Binarize: {self.binarize}")
        return img
    
    def show_image(self, image):
        cv2.imshow('Preprocessed Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()