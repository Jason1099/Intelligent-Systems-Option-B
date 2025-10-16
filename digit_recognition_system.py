import os
import numpy as np
import tensorflow as tf
import keras
from vt_model import create_vit_classifier, vt_model, PatchEncoder, QueryScaler
from preprocess import image_preprocessor
from segmentation import image_segmentation
import cv2


class DigitRecognitionSystem:
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path
        self.preprocessor = None
        self.segmentor = image_segmentation(crop_size=28)

    def prepare_mnist_data(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train[..., np.newaxis].astype("float32")
        x_test = x_test[..., np.newaxis].astype("float32")
    
        print(f"Training data shape: {x_train.shape}")
        print(f"Test data shape: {x_test.shape}")
        return (x_train, y_train), (x_test, y_test)

    def train_model(self, epochs=100, batch_size=128, vanilla=False):
        (x_train, y_train), (x_test, y_test) = self.prepare_mnist_data()
        self.model = create_vit_classifier(vanilla=vanilla)

        try:
            opt = keras.optimizers.Adam()
        except Exception:
            opt = keras.optimizers.Adam()

        self.model.compile(
            optimizer=opt,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
        )

        self.model.summary()

        lr_schedule = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=5, 
            min_lr=1e-6,
            verbose=1
        )
        
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=15, 
            restore_best_weights=True,
            verbose=1
        )

        history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            callbacks=[lr_schedule, early_stop],
            verbose=1
        )

        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"\nTest accuracy: {test_acc*100:.2f}%")
        return history

    def save_model(self, path='vit_mnist_model.keras'):
        if self.model:
            if not path.endswith('.keras'):
                path = path + '.keras'
            self.model.save(path)
            print(f"Model saved to {path}")

    def load_model(self, path='vit_mnist_model.keras'):
        if not path.endswith('.keras'):
            path = path + '.keras'
        self.model = keras.models.load_model(path, custom_objects={
            'vt_model': vt_model,
            'PatchEncoder': PatchEncoder,
            'QueryScaler': QueryScaler
        })
        print(f"Model loaded from {path}")

    def predict_image(self, image_path, out_dir="debug_digits"):
        if self.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")

        os.makedirs(out_dir, exist_ok=True)

        preprocessor = image_preprocessor(image_path=image_path, binarize=False)
        preprocessed = preprocessor.preprocess()

        bboxes, crops = self.segmentor.segmentation(preprocessed)
        if len(crops) == 0:
            print("No digits found in the image")
            return []

        # Save cropped digits
        for i, crop in enumerate(crops):
            cv2.imwrite(os.path.join(out_dir, f"digit_{i:03}.png"), crop)

        vis = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
        for i, (x, y, w, h, area) in enumerate(bboxes):
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis, str(i), (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imwrite(os.path.join(out_dir, "debug_boxes.png"), vis)

        _, debug_binary = cv2.threshold(preprocessed, 0, 255,
                                        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cv2.imwrite(os.path.join(out_dir, "debug_binary.png"), debug_binary)

        processed_crops = []
        for crop in crops:
            crop_normalized = crop.astype('float32')
            crop_reshaped = crop_normalized[..., np.newaxis]
            processed_crops.append(crop_reshaped)

        processed_crops = np.array(processed_crops)

        #Predict
        predictions = self.model.predict(processed_crops, verbose=0)
        predicted_digits = np.argmax(predictions, axis=1)
        confidence_scores = np.max(tf.nn.softmax(predictions), axis=1)

        results = []
        for i, (bbox, digit, conf) in enumerate(zip(bboxes, predicted_digits, confidence_scores)):
            results.append({
                'digit': int(digit),
                'confidence': float(conf.numpy()) if hasattr(conf, 'numpy') else float(conf),
                'bbox': tuple(map(int, bbox[:4])),
                'position': int(i)
            })

        return results


    def display_predictions(self, image_path, results):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read image: {image_path}")
            return
        for result in results:
            x, y, w, h = result['bbox']
            digit = result['digit']
            conf = result['confidence']
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{digit} ({conf:.2f})"
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow('Predictions', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()