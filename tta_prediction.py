# -*- coding: utf-8 -*-
"""
 TTA: Test Time Augmentation
"""


from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
import numpy as np

from cifar10_dataset import CIFAR10Dataset

def tta(model,test_size,generator,batch_size ,epochs = 10):
    #test_time_augmentation
    #batch_sizeは，test_sizeの約数でないといけない．
    pred = np.zeros(shape = (test_size,10), dtype = float)
    step_per_epoch = test_size //batch_size
    for epoch in range(epochs):
        print( 'epoch: ' + str(epoch+1)+'/'+str(epochs))
        for step in range(step_per_epoch):
            print( 'step: ' + str(step+1)+'/'+str(step_per_epoch))
            sta = batch_size * step
            end = sta + batch_size
            tmp_x = generator.__next__()
            pred[sta:end] += model.predict(tmp_x)
        
    return pred / epochs


def tta_generator(x_test,batch_size):
    return ImageDataGenerator(
                rotation_range = 20,
                horizontal_flip = True,
                height_shift_range = 0.2,
                width_shift_range = 0.2,
                zoom_range = 0.2,
                channel_shift_range = 0.2
            ).flow(x_test,batch_size = batch_size,shuffle = False)
    

if __name__ == '__main__':
    
    # create dataset
    dataset = CIFAR10Dataset()
    x_train, y_train, x_test, y_test = dataset.get_batch()
    
    batch_size = 2500
    tta_epochs = 2
    model = load_model("./Models/model_file.hdf5") #学習率減衰を使用して学習したモデルをロード
    tta_pred = tta(model,x_test.shape[0],tta_generator(x_test,batch_size),batch_size, epochs = tta_epochs)

    print("tta_acc: ",end = "")
    print( accuracy_score( np.argmax(tta_pred,axis = 1) , np.argmax(y_test,axis = 1)))
    