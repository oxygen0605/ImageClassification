# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 11:04:33 2019

@author: oxygen0605
"""
from keras.models import Sequential, Model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Dense, BatchNormalization
from keras.layers import Input
from keras.layers.core import Activation, Flatten
from keras import regularizers

# for MNIST
def dnn(input_shape, num_classes=10, N_HIDDEN=128, dorp_rate=0.25):
    model = Sequential()
    model.add(Dense(N_HIDDEN, input_shape=input_shape,
    				kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation('relu'))
    model.add(Dropout(dorp_rate))
    model.add(Dense(N_HIDDEN,kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation('relu'))
    model.add(Dropout(dorp_rate))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

def cnn(input_shape, num_classes):
    
	inputs = Input(shape = (32,32,3))
	x = Conv2D(64,(3,3),padding = "SAME",activation= "relu")(inputs)
	x = Conv2D(64,(3,3),padding = "SAME",activation= "relu")(x)
	x = Dropout(0.25)(x)
	x = MaxPooling2D()(x)
	
	x = Conv2D(128,(3,3),padding = "SAME",activation= "relu")(x)
	x = Conv2D(128,(3,3),padding = "SAME",activation= "relu")(x)
	x = Dropout(0.25)(x)
	x = MaxPooling2D()(x)
	
	x = Conv2D(256,(3,3),padding = "SAME",activation= "relu")(x)
	x = Conv2D(256,(3,3),padding = "SAME",activation= "relu")(x)
	x = GlobalAveragePooling2D()(x)
	
	x = Dense(1024,activation = "relu")(x)
	x = Dropout(0.25)(x)
	y = Dense(10,activation = "softmax")(x)

	return Model(input = inputs, output = y)

def deep_cnn(input_shape, num_classes):
    inputs = Input(shape = (32,32,3))
    """
    x = Conv2D(32,(3,3),padding = "SAME",activation= "relu")(inputs)
    x = Conv2D(32,(3,3),padding = "SAME",activation= "relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(32,(3,3),padding = "SAME",activation= "relu")(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.25)(x)
    """
    x = Conv2D(64,(3,3),padding = "SAME",activation= "relu")(inputs)
    x = Conv2D(64,(3,3),padding = "SAME",activation= "relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(64,(3,3),padding = "SAME",activation= "relu")(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128,(3,3),padding = "SAME",activation= "relu")(x)
    x = Conv2D(128,(3,3),padding = "SAME",activation= "relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(128,(3,3),padding = "SAME",activation= "relu")(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.25)(x)

    x = Conv2D(256,(3,3),padding = "SAME",activation= "relu")(x)
    x = Conv2D(256,(3,3),padding = "SAME",activation= "relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(256,(3,3),padding = "SAME",activation= "relu")(x)
    x = Conv2D(256,(3,3),padding = "SAME",activation= "relu")(x)
    x = Conv2D(256,(3,3),padding = "SAME",activation= "relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(512,(3,3),padding = "SAME",activation= "relu")(x)
    x = Conv2D(512,(3,3),padding = "SAME",activation= "relu")(x)
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024,activation = "relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024,activation = "relu")(x)
    x = Dropout(0.5)(x)
    y  = Dense(10,activation = "softmax")(x)

    return Model(input = inputs, output = y)


