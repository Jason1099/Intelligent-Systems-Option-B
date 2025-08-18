from preprocess import image_preprocessor

processor = image_preprocessor(image_path = 'test.png', size = (28,28))
processor.show_image()
