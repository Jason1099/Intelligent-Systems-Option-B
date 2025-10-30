import numpy as np
import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import Input
from keras.models import Sequential
from keras.utils import image_dataset_from_directory as load_image_dir
from tensorflow.keras import layers, models
from keras import callbacks
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import sklearn.model_selection
import cv2

_raw_label = {
    0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',
    10:'dot',11:'minus',12:'plus',13:'w',14:'x',15:'y',16:'z',17:'slash',18:'equals'
}
_fix = {"dot":"*"}
INV_LABELS = {k: _fix.get(v, v) for k, v in _raw_label.items()}

class CNN_Extension: 
    def __init__(self, save_path = './Models/SavedModels/CNN.keras', input_shape = (), image_path = './digits', history_path = './Models/History/history_CNN_Ext.csv', dataset_path = './symbols'):
        self.datasetPath = dataset_path 
        self.savePath = save_path
        self.imagePath = image_path
        self.historyPath = history_path
        self.labels = {
            '0': 0, '1': 1, 
            '2': 2, '3': 3, 
            '4': 4, '5': 5, 
            '6': 6, '7': 7, 
            '8': 8, '9': 9, 
            'dot': 10, 'minus': 11, 
            'plus': 12, 'w': 13, 
            'x': 14, 'y': 15, 
            'z': 16, 'slash': 17, 
            'equals': 18
        }
    
        self.y_train = None
        self.x_train = None
        self.x_test = None
        self.y_test = None
        # self.dataset_load()
        if os.path.exists(self.savePath):
            self.model = load_model(self.savePath)
        else:
            self.model = self.train_model()


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
        dataset = np.array(dataset).astype('float32') / 255


        # (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()

        # x_train_mnist = x_train_mnist.reshape(-1, 28, 28, 1).astype('float32') / 255
        # x_test_mnist = x_test_mnist.reshape(-1, 28, 28, 1).astype('float32') / 255
        
        # idx = np.random.choice(len(x_train_mnist), 5000, replace=False)
        # x_train_mnist = x_train_mnist[idx]
        # y_train_mnist = y_train_mnist[idx]

        # y_train_mnist = keras.utils.to_categorical(y_train_mnist, len(self.labels))
        # y_test_mnist = keras.utils.to_categorical(y_test_mnist, len(self.labels))   

        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        dataset, labels, test_size=0.2)

        # self.x_train = np.concatenate([x_train_mnist, x_train])
        # self.y_train = np.concatenate([y_train_mnist, y_train])

        # self.x_test = np.concatenate([x_test_mnist, x_test])
        # self.y_test = np.concatenate([y_test_mnist, y_test])

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def train_model(self):
        num_class = len(self.labels)

        model  = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28, 28, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_class, activation='softmax'))

        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
       
        earlyStop = callbacks.EarlyStopping(monitor = "val_loss", mode = "min", patience = 5, restore_best_weights = True)
       
        history = model.fit(self.x_train, self.y_train, epochs = 200, batch_size = 16, validation_data=(self.x_test, self.y_test), callbacks = [earlyStop], verbose =1)
                                
        hist_df = pd.DataFrame(history.history) 
       
        hist_csv_file = self.historyPath
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)

        model.save(self.savePath)
        return model
    
    def tiny_cnn(num_classes: int):
        inp = layers.Input(shape=(28, 28, 1), name="input_28x28x1")

        x = layers.Conv2D(32, 3, padding="same", activation="relu", name="backbone_conv1")(inp)
        x = layers.MaxPool2D(pool_size=2, name="backbone_pool1")(x)          
        x = layers.Conv2D(64, 3, padding="same", activation="relu", name="backbone_conv2")(x)
        x = layers.MaxPool2D(pool_size=2, name="backbone_pool2")(x)         
        x = layers.Conv2D(128, 3, padding="same", activation="relu", name="backbone_conv3")(x)
        x = layers.GlobalAveragePooling2D(name="backbone_gap")(x)
        x = layers.Dropout(0.25, name="head_dropout")(x)
        out = layers.Dense(num_classes, activation="softmax", name="head_logits")(x)

        model = models.Model(inp, out, name="TinyCNN")
        return model
        
    
    def predict_image(self):

        dataset = load_image_dir(self.imagePath, labels = None, color_mode = "grayscale", image_size = (28,28), shuffle = False)
        dataset = dataset.map(lambda x: x / 255.0)
        predictions = self.model.predict(dataset)
        inv_labels = {v: k for k, v in self.labels.items()}
        for x in predictions:
            print(inv_labels[np.argmax(x)])

    def load_cnn_ext(self):
        return self.model

# cnn = CNN_Extension(save_path = './Models/SavedModels/CNN_Ext_1.keras', history_path = './Models/History/history_CNN_Ext_1.csv')
# cnn.predict()