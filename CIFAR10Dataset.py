# -*- coding: utf-8 -*-
"""
@author: ozon0
"""

import keras
from keras.datasets import cifar10

class CIFAR10Dataset():
	def __init__(self):
		self.image_shape = (32, 32, 3)
		self.num_classes = 10
		
	def preprocess(self, data, label_data=False):
		if label_data:
			# conver class number to one-hot vector
			data = keras.utils.to_categorical(data, self.num_classes)
		
		else:
			data = data.astype("float32")
			data /= 255 #convert the value to 0 ~ 1 scale
			shape = (data.shape[0],) + self.image_shape
			data = data.reshape(shape)
			
		return data
	
	def get_batch(self):
		# x: data, y: lebel
		(x_train, y_train), (x_test, y_test) = cifar10.load_data()
		
		x_train, x_test = [self.preprocess(d) for d in [x_train, x_test]]
		y_train, y_test = [self.preprocess(d, label_data=True) for d in
					 [y_train, y_test]]
		
		return x_train, y_train, x_test, y_test