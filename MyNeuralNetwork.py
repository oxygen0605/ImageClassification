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
from keras.applications.vgg16 import VGG16
from keras import regularizers


# for MNIST
def dnn(input_shape=(28*28*1,), num_classes=10, num_hidden=128, dorp_rate=0.25):
    model = Sequential()
    model.add(Dense(num_hidden, input_shape=input_shape,
    				kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation('relu'))
    model.add(Dropout(dorp_rate))
    model.add(Dense(num_hidden,kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation('relu'))
    model.add(Dropout(dorp_rate))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

def cnn(input_shape, num_classes):
    
	inputs = Input(shape = input_shape)
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
	y = Dense(num_classes, activation = "softmax")(x)

	return Model(input = inputs, output = y)

def deep_cnn(input_shape, num_classes):
    inputs = Input(shape = input_shape)
    
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
    y  = Dense(num_classes, activation = "softmax")(x)

    return Model(input = inputs, output = y)

def vgg16_for_cifar10(input_shape, num_classes=10):
    
    base_model=VGG16(weights='imagenet',include_top=False,
                 input_tensor=Input(shape=(32,32,3)))

    # Disassemble layers
    layers = [l for l in base_model.layers]
    
    inputs = layers[0].output
    x = layers[1].output
    for i in range(2, 14):
        layers[i].trainable = False
        x = layers[i](x)
    
    #x = layers[11](x)
    #x = layers[12](x)
    #x = BatchNormalization()(x)
    #x = layers[13](x)
    #x = Dropout(0.25)(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024,activation = "relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Dense(1024,activation = "relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    y = Dense(10, activation = "softmax")(x)
    
    return Model(input = inputs, output = y)

def vgg16_family_cnn(input_shape, num_classes):
    
    input_layer = Input(shape=input_shape)
    
    # Block 1
    conv1_1 = Conv2D(64, (3, 3),name='conv1_1', activation='relu', padding='same')(input_layer)
    conv1_2 = Conv2D(64, (3, 3),name='conv1_2', activation='relu', padding='same')(conv1_1)
    bn1 = BatchNormalization(axis=3)(conv1_2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)
    drop1 = Dropout(0.5)(pool1)

    # Block 2
    conv2_1 = Conv2D(128, (3, 3),name='conv2_1', activation='relu', padding='same')(drop1)
    conv2_2 = Conv2D(128, (3, 3),name='conv2_2', activation='relu', padding='same')(conv2_1)
    bn2 = BatchNormalization(axis=3)(conv2_2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)
    drop2 = Dropout(0.5)(pool2)

    # Block 3
    conv3_1 = Conv2D(256, (3, 3),name='conv3_1', activation='relu', padding='same')(drop2)
    conv3_2 = Conv2D(256, (3, 3),name='conv3_2', activation='relu', padding='same')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3),name='conv3_3', activation='relu', padding='same')(conv3_2)
    bn3 = BatchNormalization(axis=3)(conv3_3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)
    drop3 = Dropout(0.5)(pool3)

    # Block 4
    conv4_1 = Conv2D(512, (3, 3),name='conv4_1', activation='relu', padding='same')(drop3)
    conv4_2 = Conv2D(512, (3, 3),name='conv4_2', activation='relu', padding='same')(conv4_1)
    conv4_3 = Conv2D(512, (3, 3),name='conv4_3', activation='relu', padding='same')(conv4_2)
    bn4 = BatchNormalization(axis=3)(conv4_3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)
    drop4 = Dropout(0.5)(pool4)

    # Block 5
    conv5_1 = Conv2D(512, (3, 3),name='conv5_1', activation='relu', padding='same')(drop4)
    conv5_2 = Conv2D(512, (3, 3),name='conv5_2', activation='relu', padding='same')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3),name='conv5_3', activation='relu', padding='same')(conv5_2)
    bn5 = BatchNormalization(axis=3)(conv5_3)
    pool5 = MaxPooling2D(pool_size=(2, 2))(bn5)
    drop5 = Dropout(0.5)(pool5)
    
    x = Flatten()(drop5)
    x = Dense(4096)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes)(x)
    x = Activation('softmax')(x)

    return Model(inputs=input_layer, outputs=x)


def vgg19_family_cnn(input_shape, num_classes):
    
    input_layer = Input(shape=input_shape)
    
    # Block 1
    conv1_1 = Conv2D(64, (3, 3),name='conv1_1', activation='relu', padding='same')(input_layer)
    conv1_2 = Conv2D(64, (3, 3),name='conv1_2', activation='relu', padding='same')(conv1_1)
    bn1 = BatchNormalization(axis=3)(conv1_2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)
    drop1 = Dropout(0.5)(pool1)

    # Block 2
    conv2_1 = Conv2D(128, (3, 3),name='conv2_1', activation='relu', padding='same')(drop1)
    conv2_2 = Conv2D(128, (3, 3),name='conv2_2', activation='relu', padding='same')(conv2_1)
    bn2 = BatchNormalization(axis=3)(conv2_2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)
    drop2 = Dropout(0.5)(pool2)

    # Block 3
    conv3_1 = Conv2D(256, (3, 3),name='conv3_1', activation='relu', padding='same')(drop2)
    conv3_2 = Conv2D(256, (3, 3),name='conv3_2', activation='relu', padding='same')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3),name='conv3_3', activation='relu', padding='same')(conv3_2)
    conv3_4 = Conv2D(256, (3, 3),name='conv3_4', activation='relu', padding='same')(conv3_3)
    bn3 = BatchNormalization(axis=3)(conv3_4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)
    drop3 = Dropout(0.5)(pool3)

    # Block 4
    conv4_1 = Conv2D(512, (3, 3),name='conv4_1', activation='relu', padding='same')(drop3)
    conv4_2 = Conv2D(512, (3, 3),name='conv4_2', activation='relu', padding='same')(conv4_1)
    conv4_3 = Conv2D(512, (3, 3),name='conv4_3', activation='relu', padding='same')(conv4_2)
    conv4_4 = Conv2D(512, (3, 3),name='conv4_4', activation='relu', padding='same')(conv4_3)
    bn4 = BatchNormalization(axis=3)(conv4_4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)
    drop4 = Dropout(0.5)(pool4)

    # Block 5
    conv5_1 = Conv2D(512, (3, 3),name='conv5_1', activation='relu', padding='same')(drop4)
    conv5_2 = Conv2D(512, (3, 3),name='conv5_2', activation='relu', padding='same')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3),name='conv5_3', activation='relu', padding='same')(conv5_2)
    conv5_4 = Conv2D(512, (3, 3),name='conv5_4', activation='relu', padding='same')(conv5_3)
    bn5 = BatchNormalization(axis=3)(conv5_4)
    pool5 = MaxPooling2D(pool_size=(2, 2))(bn5)
    drop5 = Dropout(0.5)(pool5)
    
    x = Flatten()(drop5)
    x = Dense(4096)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes)(x)
    x = Activation('softmax')(x)

    return Model(inputs=input_layer, outputs=x)
