import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import Input
from keras.models import Sequential
from keras.utils import image_dataset_from_directory as load_image_dir
import math
from keras import callbacks
import pandas as pd
import os



class CNN: 
    def __init__(self, save_path = './Models/SavedModels/CNN.keras', input_shape = (), image_path = './digits'):
        self.dataset_load()
        self.savePath = save_path
        self.imagePath = image_path

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
    
    def predict(self):

        dataset = load_image_dir(self.imagePath, labels = None, color_mode = "grayscale", image_size = (28,28), shuffle = False)
        predictions = self.model.predict(dataset)
        for x in predictions:
            print(np.argmax(x))
    

cnn = CNN()
cnn.predict()