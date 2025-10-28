import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import Input
from keras.models import Sequential
from keras import callbacks
import pandas as pd
import os
from Models.helpers.segmentation import image_segmentation 
from Models.helpers.preprocess import image_preprocessor
import cv2
import json

INV_LABELS = {i: str(i) for i in range(10)}

class CNN: 
    def __init__(self, save_path = './Models/SavedModels/CNN.keras', input_shape = (28, 28, 1), image_path = './digits'):
        self.dataset_load()
        self.savePath = save_path
        self.imagePath = image_path
        self.input_shape = input_shape
        self.segmentor = image_segmentation(crop_size=28)

        if os.path.exists(self.savePath):
            self.model = load_model(self.savePath)
        else:
            self.model = self.train_model()


    def dataset_load(self):
        (X_train, self.y_train), (X_test, self.y_test) = mnist.load_data()

        self.X_train, self.X_test = X_train/255, X_test/255
        print(self.X_train.shape)

        self.X_train = self.X_train.reshape(self.X_train.shape[0], 28, 28, 1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], 28, 28, 1)
        # return data
    def train_model(self):
    
        model  = Sequential()
        model.add(Input(shape = (28, 28, 1), batch_size = 16))
        model.add(Conv2D(32,kernel_size = (3,3), activation = 'relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(64, activation = 'relu'))
        model.add(Dense(10, activation = 'softmax'))

        model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
       
        earlyStop = callbacks.EarlyStopping(monitor = "val_loss", mode = "min", patience = 5, restore_best_weights = True)
       
        history = model.fit(self.X_train, self.y_train, epochs = 200, batch_size = 16, validation_data=(self.X_test, self.y_test), callbacks = [earlyStop], verbose = 1)
                                
        hist_df = pd.DataFrame(history.history) 
       
        hist_csv_file = './Models/History/history_CNN.csv'
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)

        model.save(self.savePath)
        return model


    def predict_image(self, image_path, out_dir="./digits_export"):
        if self.model is None:
            raise ValueError("Model not loaded")

        if os.path.exists(out_dir):
            contents = os.listdir(out_dir)
            for item in contents:
                item_path = os.path.join(out_dir, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
        dir2 = os.path.join(out_dir, "objects")
        if os.path.exists(dir2):
            contents = os.listdir(dir2)
            for item in contents:
                item_path = os.path.join(dir2, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
        os.makedirs(out_dir, exist_ok=True)

        preprocessor = image_preprocessor(image_path=image_path, binarize=False)
        preprocessed = preprocessor.preprocess()

        bboxes, crops, manifest = self.segmentor.segmentation(preprocessed)
        if len(crops) == 0:
            print("No digits")
            return []

        H, W = self.input_shape[0], self.input_shape[1]
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
        cls_idx = np.argmax(preds, axis=1)

     
        ro = manifest.get("reading_order", list(range(len(bboxes))))

        cid_to_prediction = {
            cid: str(cls_idx[pos]) 
            for pos, cid in enumerate(ro)
        }

        ordered_results = []
        for pos, cid in enumerate(ro):
            bbox = manifest["components"][cid]["bbox"]
            # --- FIX: Use the map to get the correct prediction ---
            k = cid_to_prediction[cid] 
            ordered_results.append({
                'digit': str(k),
                'bbox': tuple(map(int, bbox[:4])),
                'position': pos
            })

        with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump({
                "image_path": image_path,
                "results": ordered_results
            }, f, indent =2)
            

        # --- 4. Process Groups ---
        group_combined_numbers = {}
        for group in manifest['groups']:
            group_id = group['group_id']
            members = group['members'] # Members are correctly ordered within the group
            
            # Concatenate the digits using our new map
            combined_number = "".join(cid_to_prediction.get(cid, '') for cid in members)
            
            if combined_number:
                group_combined_numbers[group_id] = combined_number

        # --- 5. Construct Final Ordered List ---
        final_numbers = []
        processed_group_ids = set()

        # Create a quick lookup for component data
        component_lookup = {c['component_id']: c for c in manifest['components']}

        for component_id in manifest['reading_order']:
            component_data = component_lookup.get(component_id)
            
            if component_data is None:
                continue 
                
            group_id = component_data.get('group_id')
            
            if group_id is not None and group_id not in processed_group_ids:
                combined = group_combined_numbers.get(group_id)
                if combined:
                    final_numbers.append(combined)
                    processed_group_ids.add(group_id)
                
            elif group_id is None:
                digit = cid_to_prediction.get(component_id) # Use our new map
                if digit:
                    final_numbers.append(digit)
                
        print("Final list of numbers (based on direct mapping and reading order):")
        print(final_numbers)
        # Return both for compatibility
        return ordered_results, final_numbers
    
    def load_cnn(self): 
        return self.model


# cnn = CNN(save_path = './Models/SavedModels/CNN.keras')
# cnn.predict_image(image_path = './Input_Images/image.png')
