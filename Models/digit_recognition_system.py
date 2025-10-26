import os, json
import numpy as np
import tensorflow as tf
import keras
from Models.vt_model import create_vit_classifier, vt_model, PatchEncoder, QueryScaler
from Models.helpers.preprocess import image_preprocessor
from Models.helpers.segmentation import image_segmentation
from os import listdir
from os.path import isfile, join
import sklearn
import sklearn.model_selection
import cv2
import pandas as pd
import ast
import operator as op


class DigitRecognitionSystem:
    def __init__(self, model_path=None, dataset_path = './symbols'):
        self.model = None
        self.model_path = model_path
        self.preprocessor = None
        self.datasetPath = dataset_path 
        self.segmentor = image_segmentation(crop_size=28)
        self.labels = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'dot': 10, 'minus': 11, 'plus': 12, 'w': 13, 'x': 14, 'y': 15, 'z': 16, 'slash': 17, 'equals': 18}


    def prepare_mnist_data(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train[..., np.newaxis].astype("float32")
        x_test = x_test[..., np.newaxis].astype("float32")
    
        print(f"Training data shape: {x_train.shape}")
        print(f"Test data shape: {x_test.shape}")
        return (x_train, y_train), (x_test, y_test)

    def dataset_load(self):
        def get_files(path):
            return [f for f in listdir(path) if isfile(join(path, f))]


        def get_files_labels(files):
            return [self.labels[f[0:f.index('-')]] for f in files]
        
        def convert(path, filename):
            image = cv2.imread(join(path, filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            image = 255 - image
            image = np.reshape(image, (28, 28, -1))
            image = np.reshape(image, (28, 28, 1)).astype('float32')
            return image

        
        files = get_files(self.datasetPath)
        print(len(files))
        labels = get_files_labels(files)
        print(len(labels))
        labels = keras.utils.to_categorical(labels, len(self.labels))
     
        dataset = [convert(self.datasetPath, file) for file in files]
        dataset = np.array(dataset).astype('float32')

        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        dataset, labels, test_size=0.2)
        return (x_train, y_train), (x_test, y_test)

    def train_model(self, epochs=150, batch_size=128, vanilla=False):
        (x_train, y_train), (x_test, y_test) = self.dataset_load()
        self.model = create_vit_classifier(vanilla=vanilla)

        try:
            opt = keras.optimizers.Adam()
        except Exception:
            opt = keras.optimizers.Adam()

        self.model.compile(
            optimizer=opt,
            loss=keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.CategoricalAccuracy(name='accuracy')]
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
        
        hist_df = pd.DataFrame(history.history) 

        with open('./Models/History/vt_ext_2', mode='w') as f:
            hist_df.to_csv(f)

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

    def _predict_image(self, image_path, out_dir="digits_export"):
        if self.model is None:
            raise ValueError("Model not loaded")

        os.makedirs(out_dir, exist_ok=True)

        preprocessor = image_preprocessor(image_path=image_path, binarize=False)
        preprocessed = preprocessor.preprocess()

        bboxes, crops, manifest = self.segmentor.segmentation(preprocessed)
        if len(crops) == 0:
            print("No digits")
            return []

        H, W = self.model.input_shape[1], self.model.input_shape[2]
        processed_crops = []
        for crop in crops:
            if crop.ndim == 3:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            crop = cv2.resize(crop, (W, H))
            crop = crop.astype('float32') / 255
            crop = crop[..., np.newaxis]
            processed_crops.append(crop)

        X = np.stack(processed_crops, axis=0)
        preds = self.model.predict(X, verbose=0)
        probs = tf.nn.softmax(preds).numpy()
        cls_idx = np.argmax(probs, axis=1)
        confs = probs[np.arange(len(probs)), cls_idx]

        inv_labels = {v: k for k, v in self.labels.items()}

        # #Predict
        # predictions = self.model.predict(processed_crops, verbose=0)
        # predicted_digits = np.argmax(predictions, axis=1)
        # confidence_scores = np.max(tf.nn.softmax(predictions), axis=1)
        # inv_labels = {v: k for k, v in self.labels.items()}
     
        ro = manifest.get("reading_order", list(range(len(bboxes))))
        ordered_results = []
        for pos, cid in enumerate(ro):
            bbox = manifest["components"][cid]["bbox"]
            k = cls_idx[cid]
            conf = confs[cid]
            ordered_results.append({
                'digit': str(inv_labels.get(int(k), int(k))),
                'confidence': float(conf),
                'bbox': tuple(map(int, bbox[:4])),
                'position': pos
            })

        expression = "".join(r["digit"] for r in ordered_results)

        def _evaluate(exp):
            OPS = {
                ast.Add: op.add,
                ast.Sub: op.sub,
                ast.Mult: op.mul,
                ast.Div: op.truediv,
                ast.USub: op.neg
            }

            def _eval(node):
                if isinstance(node, ast.Num):
                    return node.n
                elif isinstance(node, ast.BinOp):
                    return OPS[type(node.op)](_eval(node.left), _eval(node.right))
                elif isinstance(node, ast.UnaryOp):
                    return OPS[type(node.op)](_eval(node.operand))
                else:
                    raise ValueError("Unsupported expression element")
                
            try:
                node = ast.parse(exp, mode='eval').body
                return _eval(node)
            except Exception as e:
                raise ValueError(f"Invalid expression '{exp}': {e}")
            

        result = None
        try:
            result = _evaluate(expression)
        except ValueError:
            print(f"Predicted expression (not evaluable): {expression}")

        with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump({
                "image_path": image_path,
                "expression": expression,
                "result": result,
                "results": ordered_results
            }, f, indent =2)

        return ordered_results


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

    
    