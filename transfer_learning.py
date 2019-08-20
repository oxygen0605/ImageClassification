# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 00:47:49 2019

@author: oxygen0605
"""

from keras.models import load_model
from keras.optimizers import RMSprop, Adam, Nadam

from CIFAR10Dataset import CIFAR10Dataset
from MyNeuralNetwork import vgg16_for_cifar10
from cnn_cifar10 import Trainer, Evaluator

def learning_rate_schedule_for_Adam(epoch):
	lr = 0.001
	if(epoch >= 100): lr = 0.0002 #100
	if(epoch >= 140): lr = 0.0001 #140
	return lr

# create dataset
dataset = CIFAR10Dataset()
x_train, y_train, x_test, y_test = dataset.get_batch()

model = vgg16_for_cifar10(dataset.image_shape, dataset.num_classes)
# train the model
# RMSprpの方がいいかもしれない
trainer = Trainer(model, loss="categorical_crossentropy", optimizer=Adam(),logdir = './Logs/')
#trainer.simple_train(x_train, y_train, batch_size=500, epochs=10, validation_split=0.2)
trainer.train_with_data_augmentation(
x_train, y_train, batch_size=500, epochs=20, validation_split=0.2, 
lr_scheduler=learning_rate_schedule_for_Adam)
	
# show result
evaluator = Evaluator()
score = evaluator.simple_evaluate(model, x_test, y_test)