# -*- coding: utf-8 -*-
"""
 TTA（Test Time Augmentation）クラス
 認識精度を向上させるためにテストデータに摂動を加える手法
"""


from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

class TTA:
    
    #test_time_augmentation
    #batch_sizeは，test_sizeの約数にすること
    def predict(self, model, x_test, batch_size ,epochs = 10):
        
        # Augmentation用generatorによるデータセットの作成
        data_flow = self.generator(x_test, batch_size)
        
        test_size = x_test.shape[0]
        pred = np.zeros(shape = (test_size,10), dtype = float)
        
        step_per_epoch = test_size //batch_size
        for epoch in range(epochs):
            print( 'epoch: ' + str(epoch+1)+'/'+str(epochs))
            for step in range(step_per_epoch):
                print( 'step: ' + str(step+1)+'/'+str(step_per_epoch))
                sta = batch_size * step
                end = sta + batch_size
                tmp_x = data_flow.__next__()
                pred[sta:end] += model.predict(tmp_x)        
        return pred / epochs
    
    
    def generator(self, x_test,batch_size):
        return ImageDataGenerator(
                    rotation_range = 20,
                    horizontal_flip = True,
                    height_shift_range = 0.2,
                    width_shift_range = 0.2,
                    zoom_range = 0.2,
                    channel_shift_range = 0.2
                ).flow(x_test,batch_size = batch_size,shuffle = False)
"""
    def generator(self, x_test,batch_size):
        return ImageDataGeneratorEX(
                    rotation_range = 10,
                    horizontal_flip = True,
                    height_shift_range = 0.1,
                    width_shift_range = 0.1,
                    zoom_range = 0.1,
                    channel_shift_range = 0.1,
                    #random_crop=None,
                    #mix_up_alpha=0.2,
                    #cutout_mask_size=16
                ).flow(x_test,batch_size = batch_size,shuffle = False, seed=756)
"""

if __name__ == '__main__':
    from CIFAR10Dataset import CIFAR10Dataset
    # create dataset
    dataset = CIFAR10Dataset()
    x_train, y_train, x_test, y_test = dataset.get_batch()
    
    batch_size = 2500
    tta_epochs = 2
    model = load_model('Models/cnn_model_file.hdf5') #学習率減衰を使用して学習したモデルをロード
    
    tta = TTA()
    tta_pred = tta.predict(model, x_test, batch_size, epochs = tta_epochs)
    
    from sklearn.metrics import accuracy_score
    print("tta_acc: ",end = "")
    print( accuracy_score( np.argmax(tta_pred,axis = 1) , np.argmax(y_test,axis = 1)))
    