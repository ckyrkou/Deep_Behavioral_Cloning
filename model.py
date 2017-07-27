import pickle
import numpy as np
import math
import time

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import csv

import cv2

import tensorflow as tf
tf.python.control_flow_ops = tf

from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.activations import relu, softmax
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.models import load_model

from keras.callbacks import ModelCheckpoint

from keras import backend as K


# Function to loaf the data from the csv file
def loadData():
    print("Generating Data")
    y_train=[]
    X_train =[]
    count = 0
                    
    with open('driving_log.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        next(readCSV, None)
        for line in readCSV:
            
            if(float(line[3]) == 0 and count < 2000):
                count = count + 1
                continue
            
            angle=float(line[3])
            # Add a 10%  offset to the angle to prevent understeering
            y_train.append(angle)
            img = load_img(line[0])      
            x = img_to_array(img)
            x=cv2.resize(x,(80,40))
            X_train.append(x)
            
            # If the steering angle is 0 then do not flip the image
            if(float(line[3]) == 0):
                continue
            else:
                X_train.append(cv2.flip(x,1) )
                y_train.append(-(angle))
    
    y_train = np.array(y_train)
    
    X_train = np.array(X_train)
    return X_train,y_train

# Function to normalize the input images to the CNN
def normLayer(x):
    #x = K.resize_images(x,0.5,0.5,3)
    x = (x/127.5) - 0.5
    return x

# Function that generates the CNN model
# It also adds dropouts to prevent overfitting
def generate_model():
    
    print("Constructing Model")
    model = Sequential()
    
    #Normalize input    
    model.add(Lambda(normLayer, input_shape=in_shape))

    #First Convolutional Stage
    model.add(Convolution2D(16, 5, 5, border_mode='valid'))
    model.add(Activation('relu'))
##    Maybe you can try using the activation function ELU. :)
##    Use of ELU's instead of the popular ReLU's. The benefits of ELU's over ReLU's have been published in the
##    FAST AND ACCURATE DEEP NETWORK LEARNING BY EXPONENTIAL LINEAR UNITS ELUS.

    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Dropout(0.25))
    
    #Second Convolutional Stage
    model.add(Convolution2D(32, 5, 5, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Dropout(0.25))
    
    #Third Convolutional Stage
    model.add(Convolution2D(32, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    
    model.add(Dropout(0.5))
    
    #First Fully Connected Layer
    model.add(Dense(256))
    model.add(Activation('relu'))
    
    model.add(Dropout(0.25))
    
    #Second Fully Connected Layer
    model.add(Dense(128))
    model.add(Activation('relu'))
    
    #output Neuron
    model.add(Dense(1))
    
    return model

#This function trains the regression model with the input data set
def train_model(model,X_train,y_train):
    
    print("Training Model")
    
    # Image data generator to augment the data
    datagen = ImageDataGenerator(rotation_range = 2,
                             featurewise_center = False,
                             featurewise_std_normalization=False,
                             zoom_range = [0.8, 1],
                             fill_mode = 'constant',
                             cval=0)
    
    # Setup a regression network with adam optimizer
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # Incrementally save the best model basd on loss value
    chkpnt = ModelCheckpoint('model.h5',monitor='loss',verbose=1,save_best_only=True,mode='min')
    callbacks_list = [chkpnt]

    # Shuffle data
    X_train,y_train = shuffle(X_train,y_train)
    
    #Train the network with a batch size of 32 using the image data generator for a total of 10 epochs
    model.fit_generator(datagen.flow(X_train,y_train,batch_size=64),samples_per_epoch=len(X_train),nb_epoch=10,callbacks=callbacks_list,verbose=1)
    #,save_to_dir='./AugData',save_prefix='aug'
    model.save("model_final.h5")
    return model
    


t = time.time()

X_train,y_train = loadData()

print(np.shape(X_train))
print("Done Loading the Data!")
print("Took %.3f seconds" % (time.time() - t))
print("There are a total of ",np.shape(X_train)[0]," image samples.")

#Plot Training Data
plt.hist(y_train,50)

in_shape = np.shape(X_train)[1:4]
    
model = generate_model()    
    
print(model.summary())

model = train_model(model,X_train[:,:,:,:],y_train[:])

#model = load_model("model_final.h5")

# Visualize the filter of the first convolutional layer
W1 = model.layers[1].get_weights()[0]

plt.figure(figsize=(4,4),frameon=False)

layer_num = np.shape(W1)[3]

for ind in range(layer_num):
    plt.subplot(4,4,ind+1)
    im=W1[:,:,:,ind]
    plt.axis("off")
    plt.imshow(im,interpolation='nearest')

# Check the output for 20 images
steer_angle = model.predict(X_train[70:90,:,:,:],batch_size =20)

print("True Angle | Predicted Value")

for i in range(20):
    print(y_train[i+50]," | ",float(steer_angle[i]))

plt.show()
