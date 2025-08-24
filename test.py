from preprocess import image_preprocessor

processor = image_preprocessor(image_path = 'test.png', size = (500,500), binarize=True)
processor.show_image()

