# -*- coding: utf-8 -*-
"""
@author: ozon0
"""

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img
import glob

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
	
	def save_image(self, x, y, save_dir):
		
		for i, x_i in enumerate(x):
			save_img(save_dir+'/'+str(y[i][0])+'_'+str(i)+'.jpg', array_to_img(x_i))
			
	
	def load_rescaled_image_array(self, image_dir=None):
		# cifar-10用画像ロード関数
		
		if image_dir is None:
			print("Couldn't load images. Please set directory path.")

		
		x = []
		y = []
		for picture in glob.glob(image_dir+'*.jpg'):
		    img = img_to_array(load_img(picture, target_size=(32,32)))
		    x.append(img)
		    y.append(0)
		
		#正しく型変換する
		

if __name__ == '__main__':
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	dataset = CIFAR10Dataset()
	#dataset.save_image(x_test, y_test,save_dir='./Images/')
	x_train