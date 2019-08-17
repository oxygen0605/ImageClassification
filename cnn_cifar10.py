# -*- coding: utf-8 -*-
"""
Image Classification of CIFAR-10 on keras
@author: oxygen0605
"""

import os
from keras.models import load_model
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import RMSprop, Adam, Nadam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
import numpy as np

# 自作ライブラリ
import MyNeuralNetwork
from CIFAR10Dataset import CIFAR10Dataset
from MakeTensorboard import make_tensorboard
from TTAPrediction import TTA


class Trainer():
	
	def __init__(self, model, loss, optimizer, logdir = './Logs/logdir_cifar10_cnn/'):
		self._target = model
		self._target.compile(
				loss=loss, optimizer=optimizer, metrics=["accuracy"]
				)
		self.verbose = 1 # visualize progress bar: 0(OFF), 1(On), 2(On:each data) 
		self.log_dir = os.path.join(os.path.dirname(__file__), logdir)
		self.model_file_name = "model_file.hdf5"

	def simple_train(self, x_train, y_train, batch_size, epochs, validation_split):
		"""
        # remove previous execution
		if os.path.exists(self.log_dir):
			import shutil
			shutil.rmtree(self.log_dir) 
		os.mkdir(self.log_dir)
        """
        
		model_path = os.path.join(self.log_dir, self.model_file_name)
		self._target.fit(
			x_train, y_train,
			batch_size=batch_size, epochs=epochs,
			validation_split=validation_split,
			callbacks=[
				TensorBoard(log_dir=self.log_dir),
				ModelCheckpoint(model_path, save_best_only=True)
			],
			verbose=self.verboseevaluate
        )
	
	def train_with_data_augmentation(self, 
            x_train, y_train, batch_size, epochs, validation_split, lr_scheduler):
		"""
        fit(ImageDataGeneratorクラス)
        与えられたサンプルデータに基づいて，データに依存する統計量を計算します． 
        featurewise_center，featurewise_std_normalization，
        または，zca_whiteningが指定されたときに必要です．
        
        fit_generator(Modelクラス)
         Pythonジェネレータ（またはSequenceのインスタンス）によりバッチ毎に生成されたデータでモデルを訓練します．
         本ジェネレータは効率性のためモデルに並列して実行されます．例えば，モデルをGPUで学習させながら
         CPU上で画像のリアルタイムデータ拡張を行うことができるようになります．
        """
		"""
		# remove previous execution
		if os.path.exists(self.log_dir):
			import shutil
			shutil.rmtree(self.log_dir) 
		os.mkdir(self.log_dir)
		"""
        
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
		
        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
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
		model_path = os.path.join(self.log_dir, self.model_file_name)
		self._target.fit_generator(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            steps_per_epoch=x_train.shape[0] // batch_size,
            epochs=epochs,
            validation_data=(x_valid, y_valid),
            callbacks=[
                LearningRateScheduler(lr_scheduler),
                make_tensorboard(set_dir_name=self.log_dir),
                ModelCheckpoint(model_path, save_best_only=True)
            ],
            verbose=self.verbose,
            workers=4
        )
		

class Evaluator():
    
    def __init__(self, result_file_path="./prediction_result.csv"):
        self.result_file_path="./prediction_result.csv"
        
    def simple_evaluate(self, model, x_test, label):
        print("start evaluation...")
        score = model.evaluate(x_test, y_test, verbose=1)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])
    
    def tta_evaluate(self, model, x_test, batch_size = 2500, tta_epochs = 2):
        print("batch size (TTA): "+str(batch_size))
        print("epochs (TTA): "+str(tta_epochs))
        tta = TTA()
        tta_pred = tta.predict(model, x_test, batch_size, epochs = tta_epochs)
        print("Test accuracy(TTA): ",end = "")
        print( accuracy_score( np.argmax(tta_pred,axis = 1) , np.argmax(y_test,axis = 1)))    


def learning_rate_schedule_for_Adam(epoch):
	lr = 0.001
	if(epoch >= 100):
		lr/=5
	if(epoch>=140):
		lr/=2
	return lr

if __name__ == '__main__':
	
	# create dataset
	dataset = CIFAR10Dataset()
	x_train, y_train, x_test, y_test = dataset.get_batch()
	
	# create model
	#model = MyNeuralNetwork.cnn(dataset.image_shape, dataset.num_classes)
	model = load_model('Models/model_file.hdf5')
    
	# train the model
	trainer = Trainer(model, loss="categorical_crossentropy", optimizer=Adam(), )
	#trainer.simple_train(x_train, y_train, batch_size=500, epochs=10, validation_split=0.2)
	trainer.train_with_data_augmentation(
            x_train, y_train, batch_size=500, epochs=10, validation_split=0.2, 
            lr_scheduler=learning_rate_schedule_for_Adam)
	
    # show result
	evaluator = Evaluator()
	score = evaluator.simple_evaluate(model, x_test, y_test)
	
	
	
	
	
	
	
	
	
	

		