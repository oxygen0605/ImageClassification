"""
Created on Thu Aug  8 16:56:15 2019

@author: ozon0
"""

from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from make_tensorboard import make_tensorboard

# for reproducibility
np.random.seed(1671)

# Network and Training
NB_EPOCH         = 10
BATCH_SIZE       = 128
VERBOSE          = 1
NB_CLASSES       = 10
OPTIMIZER        = SGD()
N_HIDDEN         = 128
VALIDATION_SPLIT = 0.2

# data: shuffled and split between train and test sets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

#X_train is 60000 rows of 28x28 values --> reshaped in 60000 x 784
RESHAPED = 784
X_train = X_train.reshape(60000, RESHAPED)
X_test  = X_test.reshape(10000, RESHAPED)

# dataset changed into float32 for GPU
X_train = X_train.astype('float32')
X_test  = X_test.astype('float32')

#normalize
X_train /= 255
X_test  /= 255

print(X_train.shape[0], 'train samples')
print(X_train.shape[0], 'test samples')

Y_train = np_utils.to_categorical(Y_train, NB_CLASSES)
Y_test  = np_utils.to_categorical(Y_test,  NB_CLASSES)

# 10 outputs
# final stage is softmax
model = Sequential()
model.add(Dense(NB_CLASSES, input_shape=(RESHAPED, )))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
			  optimizer=OPTIMIZER,
			  metrics=['accuracy'])

callbacks = [make_tensorboard(set_dir_name='./Logs/')]

model.fit(X_train, Y_train,
		  batch_size=BATCH_SIZE, epochs=NB_EPOCH,
		  callbacks=callbacks, verbose=VERBOSE,
		  validation_split=VALIDATION_SPLIT)

score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print("\nTest score:" , score[0])
print('Test accuracy:', score[1])
