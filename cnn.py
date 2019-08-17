# -*- coding: utf-8 -*-
"""
Image Classification of CIFAR-10 on keras
@author: ozon0
"""

import os
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Dense, BatchNormalization
from keras.layers import Input
from keras.layers.core import Activation, Flatten
from keras.optimizers import RMSprop, Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

from cifar10_dataset import CIFAR10Dataset

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

class Trainer():
	
	def __init__(self, model, loss, optimizer, logdir = './Logs/logdir_cifar10_cnn'):
		self._target = model
		self._target.compile(
				loss=loss, optimizer=optimizer, metrics=["accuracy"]
				)
		self.verbose = 1 # visualize progress bar: 0(OFF), 1(On), 2(On:each data) 
		self.log_dir = os.path.join(os.path.dirname(__file__), logdir)
		self.model_file_name = "model_file.hdf5"
        
	def step_decay_for_Adam(self, epoch):
		lr = 0.001
		if(epoch >= 100):
			lr/=5
		if(epoch>=140):
			lr/=2
		return lr
	
	def train_with_data_augmentation(self, x_train, y_train, batch_size, epochs, validation_split):
		# remove previous execution
		if os.path.exists(self.log_dir):
			import shutil
			shutil.rmtree(self.log_dir) 
		os.mkdir(self.log_dir)
		
		datagen = ImageDataGenerator(
			featurewise_center=False,            # set input mean to 0 over the dataset
            samplewise_center=False,             # set each sample mean to 0
            featurewise_std_normalization=False, # divide inputs by std
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,                 # apply ZCA whitening
            rotation_range=20,                   # randomly rotate images in the range (0~180)
            width_shift_range=0.2,               # randomly shift images horizontally
            height_shift_range=0.2,              # randomly shift images vertically
            zoom_range = 0.2,
            channel_shift_range = 0.2,
            horizontal_flip=True,                # randomly flip images
            vertical_flip=False                  # randomly flip images
		)
        
		"""
        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        与えられたサンプルデータに基づいて，データに依存する統計量を計算します． 
        featurewise_center，featurewise_std_normalization，
        または，zca_whiteningが指定されたときに必要です．
        """
		datagen.fit(x_train)
		
        # for reproducibility
		np.random.seed(1671)
        #split for validation data
		indices = np.arange(x_train.shape[0])
		np.random.shuffle(indices)
		
		#シャッフルしたindicesの後ろからvalidataion_size分だけvalidationに回す
		validation_size = int(x_train.shape[0] * validation_split)
		x_train, x_valid = \
            x_train[indices[:-validation_size], :], \
            x_train[indices[-validation_size:], :]
		y_train, y_valid = \
            y_train[indices[:-validation_size], :], \
            y_train[indices[-validation_size:], :]
        
        # training
        # Pythonジェネレータ（またはSequenceのインスタンス）によりバッチ毎に生成されたデータでモデルを訓練します．
        # 本ジェネレータは効率性のためモデルに並列して実行されます．例えば，モデルをGPUで学習させながら
        # CPU上で画像のリアルタイムデータ拡張を行うことができるようになります．
        
		model_path = os.path.join(self.log_dir, self.model_file_name)
		self._target.fit_generator(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            steps_per_epoch=x_train.shape[0] // batch_size,
            epochs=epochs,
            validation_data=(x_valid, y_valid),
            callbacks=[
                LearningRateScheduler(self.step_decay_for_Adam),
                TensorBoard(log_dir=self.log_dir),
                ModelCheckpoint(model_path, save_best_only=True)
            ],
            verbose=self.verbose,
            workers=4
        )
		
	def simple_train(self, x_train, y_train, batch_size, epochs, validation_split):
		# remove previous execution
		if os.path.exists(self.log_dir):
			import shutil
			shutil.rmtree(self.log_dir) 
		os.mkdir(self.log_dir)
		
		model_path = os.path.join(self.log_dir, self.model_file_name)
		self._target.fit(
			x_train, y_train,
			batch_size=batch_size, epochs=epochs,
			validation_split=validation_split,
			callbacks=[
				TensorBoard(log_dir=self.log_dir),
				ModelCheckpoint(model_path, save_best_only=True)
			],
			verbose=self.verbose
		)
		
if __name__ == '__main__':
	
	# create dataset
	dataset = CIFAR10Dataset()
	x_train, y_train, x_test, y_test = dataset.get_batch()
	
	# create model
	#model = cnn(dataset.image_shape, dataset.num_classes)
	model = load_model('Models/model_file.hdf5')
    
	# train the model
	trainer = Trainer(model, loss="categorical_crossentropy", optimizer=Adam())
	#trainer.train(x_train, y_train, batch_size=500, epochs=10, validation_split=0.2)
	trainer.train_with_data_augmentation(x_train, y_train, batch_size=500, epochs=10, validation_split=0.2)
	
    # show result
	score = model.evaluate(x_test, y_test, verbose=0)
	print("Test loss:", score[0])
	print("Test accuracy:", score[1])
	
	
	
	
	
	
	
	
	
	

		