# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 11:04:33 2019

@author: oxygen0605
"""
from keras.models import Sequential, Model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers import Dropout, Dense, BatchNormalization
from keras.layers import Input, add
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

    x = Dense(1024)(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(1024)(x)
	
    x = Dropout(0.5)(x)
    x = Dense(1024)(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
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

# this is a part of ResNet and WideResNet.
def rescell(data, filters, kernel_size, option=False):
    
    strides=(1,1)
    if option:
        strides=(2,2)
    
    x=Conv2D(filters=filters,
			 kernel_size=kernel_size,
			 strides=strides,
			 padding="same",
			 kernel_initializer='he_normal')(data)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)

    # shortcut 
    data=Conv2D(filters=int(x.shape[3]), 
				kernel_size=(1,1), 
				strides=strides, 
				padding="same",
				kernel_initializer='he_normal')(data)

    x=Conv2D(filters=filters,
			 kernel_size=kernel_size,
			 strides=(1,1),
			 padding="same",
			 kernel_initializer='he_normal')(x)
    x=BatchNormalization()(x)

    # connnection
    x=add([x,data])

    x=Activation('relu')(x)

	
    return x

# this is a part of ResNet and WideResNet.
def ResBlock(data, filters, kernel_size, option=False):
    
    strides=(1,1)
    if option:
        strides=(2,2)
    
    x=BatchNormalization()(data)
    x=Activation('relu')(x)
    x=Conv2D(filters=filters,
			 kernel_size=kernel_size,
			 strides=strides,
			 padding="same",
			 kernel_initializer='he_normal')(x)

    # shortcut 
    data=Conv2D(filters=int(x.shape[3]),
				kernel_size=(1,1),
				strides=strides,
				padding="same",
				kernel_initializer='he_normal')(data)

    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    x=Dropout(0.3)(x)
    x=Conv2D(filters=filters,
			 kernel_size=kernel_size,
			 strides=(1,1),
			 padding="same",
			 kernel_initializer='he_normal')(x)

	# connection
    x=add([x,data])
    return x

def ResNet34(input_shape, num_classes):
	input=Input(shape=input_shape)
  
	x=Conv2D(32,(7,7), padding="same", activation="relu",kernel_initializer='he_normal')(input)
	x=MaxPooling2D(pool_size=(2,2))(x)

	x=rescell(x,64,(3,3))
	x=rescell(x,64,(3,3))
	x=rescell(x,64,(3,3))

	x=rescell(x,128,(3,3),True)

	x=rescell(x,128,(3,3))
	x=rescell(x,128,(3,3))
	x=rescell(x,128,(3,3))

	x=rescell(x,256,(3,3),True)

	x=rescell(x,256,(3,3))
	x=rescell(x,256,(3,3))
	x=rescell(x,256,(3,3))
	x=rescell(x,256,(3,3))
	x=rescell(x,256,(3,3))

	x=rescell(x,512,(3,3),True)

	x=rescell(x,512,(3,3))
	x=rescell(x,512,(3,3))

	x=AveragePooling2D(pool_size=(int(x.shape[1]),int(x.shape[2])),strides=(2,2))(x)

	x=Flatten()(x)
	x=Dense(units=num_classes,kernel_initializer="he_normal",activation="softmax")(x)
	model=Model(inputs=input,outputs=[x])
  
	return model

"""
 We use SGD with a mini-batch size of 256. The learning rate starts from 0.1 
and is divided by 10 when the error plateaus, 
and the models are trained for up to 60 × 104 iterations.
 We use a weight decay of 0.0001 and a momentum of 0.9.
 We do not use dropout [14], following the practice in [16].

"""

# Define WRN-28-10.
def WideResNet(input_shape, num_classes):
	input=Input(shape=input_shape)

	k = 10 # 論文によれば、CIFAR-10に最適な値は10。
	n= 4   # 論文によれば、CIFAR-10に最適な値は4。
	       # WRN-28-10の28はconvの数で、「1（入り口のconv）+ 3 * n * 2 + 3（ショートカットの中のconv？）」みたい。
		   # n = 4 で28。
	
	#conv1
	x=Conv2D(16,(3,3), padding="same", activation="relu",kernel_initializer='he_normal')(input)
	
  # conv2
	for i in range(2*n):
		x=ResBlock(x,16*k,(3,3))
	
  #conv3
	x=ResBlock(x,32*k,(3,3),True)
	for i in range(2*n-1):
		x=ResBlock(x,32*k,(3,3))
	
  #conv4
	x=ResBlock(x,64*k,(3,3),True)
	for i in range(2*n-1):
		x=ResBlock(x,64*k,(3,3))
	
	x=BatchNormalization()(x)
	x=Activation('relu')(x)
	x=GlobalAveragePooling2D()(x)
	
	x=Dense(units=num_classes,kernel_initializer="he_normal",activation="softmax")(x)
	model=Model(inputs=input,outputs=[x])
  
	return model
	
if __name__ == '__main__':
	#deep_cnn((32,32,3), 10).summary()
	ResNet((32,32,3), 10).summary()
