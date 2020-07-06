# -*- coding: utf-8 -*-
"""
画像認識プログラム。
モデル：CNN, vgg16, WideResNet
     (MyNeuralNetworkから選択可能)
学習/訓練データセット：CIFAR-10
ライブラリ：tensorflow, (tensorflow-gpu,) keras, numpy 
"""

import os
from keras.models import load_model
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import RMSprop, Adam, Nadam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
import numpy as np

# 自作ライブラリ
from Libs import MyNeuralNetwork
from Libs.CIFAR10Dataset import CIFAR10Dataset
from Libs.ImageDataGeneratorEX import ImageDataGeneratorEX
from Libs.MakeTensorboard import make_tensorboard
from Libs.TTAPrediction import TTA


class Trainer():
    
    def __init__(self, model, loss, optimizer, logdir = './Logs/logdir_cifar10_cnn/'):
        self._target = model
        self._target.compile(
                loss=loss, optimizer=optimizer, metrics=["accuracy"]
                )
        self.verbose = 1 # visualize progress bar: 0(OFF), 1(On), 2(On:each data) 
        self.log_dir = os.path.join(os.path.dirname(__file__), logdir)
        self.model_file_name = "model_file.hdf5"

    def simply_train(self, x_train, y_train, batch_size, epochs, validation_split):
        
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
    def train_for_tuning_test_data(self, 
            x_train, y_train, x_test, y_test, batch_size, epochs, lr_scheduler):
        
        datagen = ImageDataGeneratorEX(
                  featurewise_center=False,      # set input mean to 0 over the dataset
                  samplewise_center=False,             # set each sample mean to 0
                  featurewise_std_normalization=False, # divide inputs by std
                  samplewise_std_normalization=False,  # divide each input by its std
                  zca_whitening=False,                 # apply ZCA whitening
                  rotation_range=0,                    # randomly rotate images in the range (0~180)
                  width_shift_range=0.0,               # randomly shift images horizontally
                  height_shift_range=0.0,              # randomly shift images vertically
                  zoom_range = 0.0,
                  channel_shift_range = 0.0,
                  horizontal_flip=False,               # randomly flip images
                  vertical_flip=False,                 # randomly flip images
                  random_crop=None,
                  mix_up_alpha=0.2,
                  cutout_mask_size=16
        )
        
        # training (validation dataはデータ拡張はしない)
        model_path = os.path.join(self.log_dir, self.model_file_name)
        self._target.fit_generator(
            generator        = datagen.flow(x_train,y_train, batch_size,seed=0),
            steps_per_epoch  = x_train.shape[0] // batch_size,
            epochs           = epochs,
            validation_data  = ImageDataGenerator().flow(x_test,y_test, batch_size),
                  validation_steps = x_test.shape[0] // batch_size,
            callbacks=[
                LearningRateScheduler(lr_scheduler),
                make_tensorboard(set_dir_name=self.log_dir),
                ModelCheckpoint(model_path, save_best_only=True,monitor='val_acc',mode='max')
            ],
            verbose = self.verbose,
            use_multiprocessing=True,
            #workers = 4
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
        
    def train_for_tuning_test_data(self, 
            x_train, y_train, x_test, y_test, batch_size, epochs, lr_scheduler):
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
        
        # training        
        model_path = os.path.join(self.log_dir, self.model_file_name)
        self._target.fit_generator(
            generator        = datagen.flow(x_train,y_train, batch_size),
            steps_per_epoch  = x_train.shape[0] // batch_size,
            epochs           = epochs,
            validation_data  = ImageDataGenerator().flow(x_test,y_test, batch_size),
            validation_steps = x_test.shape[0] // batch_size,
            callbacks=[
                LearningRateScheduler(lr_scheduler),
                make_tensorboard(set_dir_name=self.log_dir),
                ModelCheckpoint(model_path, save_best_only=True)
            ],
            verbose = self.verbose,
            workers = 4
        )

class Evaluator():
    
    def __init__(self, result_file_path="./prediction_result.csv"):
        self.result_file_path="./prediction_result.csv"
        
    def simple_evaluate(self, model, x_test, label):
        print("start evaluation...")
        score = model.evaluate(x_test, y_test, verbose=1)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])
    
    def tta_evaluate(self, model, x_test, label, batch_size = 2500, tta_epochs = 2):
        print("batch size (TTA): "+str(batch_size))
        print("epochs (TTA): "+str(tta_epochs))
        tta = TTA()
        tta_pred = tta.predict(model, x_test, batch_size, epochs = tta_epochs)
        print("Test accuracy(TTA): ",end = "")
        print( accuracy_score( np.argmax(tta_pred,axis = 1) , np.argmax(label,axis = 1)))
        return tta_pred   


def learning_rate_schedule_for_Adam(epoch):
    lr = 0.001
    if(epoch >= 100): lr = 0.0002 #100
    if(epoch >= 140): lr = 0.0001 #140
    return lr

if __name__ == '__main__':
    
    # create dataset
    dataset = CIFAR10Dataset()
    x_train, y_train, x_test, y_test = dataset.get_batch()
    
    save_dir='./Models/'
    
    # create model
    model = MyNeuralNetwork.deep_cnn(dataset.image_shape, dataset.num_classes)
    #model = MyNeuralNetwork.vgg16_family_cnn(dataset.image_shape, dataset.num_classes)
    #model = MyNeuralNetwork.WideResNet(dataset.image_shape, dataset.num_classes)
    
    # show model spec
    model.summary()
    
    # train the model
    trainer = Trainer(model, loss="categorical_crossentropy", optimizer=Adam(), logdir=save_dir)
    trainer.train_for_tuning_test_data(
            x_train, y_train, x_test, y_test, batch_size=128, epochs=300, 
            lr_scheduler=learning_rate_schedule_for_Adam)
    
    # show result
    evaluator = Evaluator()
    score = evaluator.simple_evaluate(model, x_test, y_test)
    
    # show result
    score = evaluator.tta_evaluate(model, x_test, y_test, batch_size = 500, tta_epochs = 50)
    
    
    
    
    
    
    
    

        