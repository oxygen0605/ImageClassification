# -*- coding: utf-8 -*-
"""
@author: ozon0
"""

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

#　検証用
from CIFAR10Dataset import CIFAR10Dataset 
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.datasets import cifar10


class ImageDataGeneratorEX(ImageDataGenerator):
	def __init__(self,
               featurewise_center=False,
               samplewise_center=False, 
               featurewise_std_normalization=False,
               samplewise_std_normalization=False,
               zca_whitening=False,
               zca_epsilon=1e-06,
               rotation_range=0.0,
               width_shift_range=0.0,
               height_shift_range=0.0,
               brightness_range=None,
               shear_range=0.0,
               zoom_range=0.0, 
               channel_shift_range=0.0,
               fill_mode='nearest',
               cval=0.0,
               horizontal_flip=False, 
               vertical_flip=False,
               rescale=None,
               preprocessing_function=None,
               data_format=None,
               validation_split=0.0, 
               random_crop=None,    # a new parameter for random crop
               mix_up_alpha=0.0,    # a new parameter for mix up
               cutout_mask_size=0   # a new parameter for cutout
              ):
    
		# 親クラスのコンストラクタ
		super().__init__(featurewise_center=featurewise_center,
                     samplewise_center=samplewise_center,
                     featurewise_std_normalization=featurewise_std_normalization,
                     samplewise_std_normalization=samplewise_std_normalization,
                     zca_whitening=zca_whitening,
					 zca_epsilon=zca_epsilon,
                     rotation_range=rotation_range,
                     width_shift_range=width_shift_range,
                     height_shift_range=height_shift_range,
                     brightness_range=brightness_range,
                     shear_range=shear_range,
                     zoom_range=zoom_range,
                     channel_shift_range=channel_shift_range,
                     fill_mode=fill_mode,
                     cval=cval,
                     horizontal_flip=horizontal_flip,
                     vertical_flip=vertical_flip,
                     rescale=rescale,
                     preprocessing_function=preprocessing_function,
                     data_format=data_format,
                     validation_split=validation_split)
		
		# 拡張処理のパラメーター
		# Mix-up
		assert mix_up_alpha >= 0.0
		self.mix_up_alpha = mix_up_alpha
		# Random Crop
		assert random_crop == None or len(random_crop) == 2
		self.random_crop_size = random_crop
		self.cutout_mask_size = cutout_mask_size
    
	# ランダムクロップ
    # 参考 https://jkjung-avt.github.io/keras-image-cropping/
	def random_crop(self, original_img):
        # Note: image_data_format is 'channel_last'
		assert original_img.shape[2] == 3
		if original_img.shape[0] < self.random_crop_size[0] or original_img.shape[1] < self.random_crop_size[1]:
			raise ValueError(f"Invalid random_crop_size : original = {original_img.shape}, crop_size = {self.random_crop_size}")
		height, width = original_img.shape[0], original_img.shape[1]
		dy, dx = self.random_crop_size
		x = np.random.randint(0, width - dx + 1)
		y = np.random.randint(0, height - dy + 1)
		return original_img[y:(y+dy), x:(x+dx), :]

    # Mix-up
    # 参考 https://qiita.com/yu4u/items/70aa007346ec73b7ff05
	def mix_up(self, X1, y1, X2, y2):
		assert X1.shape[0] == y1.shape[0] == X2.shape[0] == y2.shape[0]
		batch_size = X1.shape[0]
		l = np.random.beta(self.mix_up_alpha, self.mix_up_alpha, batch_size)
		X_l = l.reshape(batch_size, 1, 1, 1)
		y_l = l.reshape(batch_size, 1)
		X = X1 * X_l + X2 * (1-X_l)
		y = y1 * y_l + y2 * (1-y_l)
		return X, y
    
	def cutout(self, x, y):
		return np.array(list(map(self._cutout, x))), y

	def _cutout(self, image_origin):
		# 最後に使うfill()は元の画像を書き換えるので、コピーしておく
		img = np.copy(image_origin)
		mask_value = img.mean()
		# 乱数固定(flowでseed固定したら必要ないかも)

		h, w, _ = img.shape
		# マスクをかける場所のtop, leftをランダムに決める
		# はみ出すことを許すので、0以上ではなく負の値もとる(最大mask_size // 2はみ出す)
		top = np.random.randint(0 - self.cutout_mask_size // 2, h - self.cutout_mask_size)
		left = np.random.randint(0 - self.cutout_mask_size // 2, w - self.cutout_mask_size)
		bottom = top + self.cutout_mask_size
		right = left + self.cutout_mask_size

		# はみ出した場合の処理
		if top < 0:
			top = 0
		if left < 0:
			left = 0

		# マスク部分の画素値を平均値で埋める
		img[top:bottom, left:right, :].fill(mask_value)
		return img


	def flow(self, 
			 x, y=None, 
			 batch_size=32, 
			 shuffle=True,
			 sample_weight=None,
			 seed=None, 
			 save_to_dir=None, 
			 save_prefix='', 
			 save_format='png', 
			 subset=None
		):
		# 親クラスのコンストラクタ
		batches = super().flow(x, y=y,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           sample_weight=sample_weight,
                           seed=seed,
                           save_to_dir=save_to_dir,
                           save_prefix=save_prefix,
                           save_format=save_format,
                           subset=subset)
		# 拡張処理
		while True:
			batch_x, batch_y = next(batches)
			
			# mix up
			if self.mix_up_alpha > 0:
				while True:
					batch_x_2, batch_y_2 = next(batches)
					m1, m2 = batch_x.shape[0], batch_x_2.shape[0]
					
					if m1 < m2:
						batch_x_2 = batch_x_2[:m1]
						batch_y_2 = batch_y_2[:m1]
						break
					elif m1 == m2:
						break
				batch_x, batch_y = self.mix_up(batch_x, batch_y, batch_x_2, batch_y_2)
			
			# Random crop
			if self.random_crop_size is not None:
				x = np.zeros((batch_x.shape[0], self.random_crop_size[0], self.random_crop_size[1], 3))
				for i in range(batch_x.shape[0]):
					x[i] = self.random_crop(batch_x[i])
				batch_x = x
			
			if self.cutout_mask_size > 0:
				batch_x, batch_y = self.cutout(batch_x, batch_y)
			
			
			yield (batch_x, batch_y)

def show_imgs(imgs, row, col):
    """Show PILimages as row*col
     # Arguments
            imgs: 1-D array, include PILimages
            row: Int, row for plt.subplot
            col: Int, column for plt.subplot
    """
    if len(imgs) != (row * col):
        raise ValueError("Invalid imgs len:{} col:{} row:{}".format(len(imgs), row, col))

    for i, img in enumerate(imgs):
        plot_num = i+1
        plt.subplot(row, col, plot_num)
        plt.tick_params(labelbottom="off") # x軸の削除
        plt.tick_params(labelleft="off") # y軸の削除
        plt.imshow(img)
    plt.show()

if __name__ == '__main__':
	
	dataset = CIFAR10Dataset()
	x_train, y_train, x_test, y_test = dataset.get_batch()
	
	ex_gen = ImageDataGeneratorEX(
			   rotation_range=0,
			   horizontal_flip=True,
			   zoom_range=0.0,
			   random_crop=None,
			   mix_up_alpha=0.0, 
			   cutout_mask_size=16
		     )
	
	gen = ImageDataGenerator(
			   width_shift_range=4,
			   rotation_range=0,
			   horizontal_flip=True,
			   zoom_range=0.0,
		     )
	
	max_img_num = 16
	imgs = []
	for d in ex_gen.flow(x_train, y_train, batch_size=1, seed = 0):
	    # このあと画像を表示するためにndarrayをPIL形式に変換して保存する
	    imgs.append(image.array_to_img(d[0][0], scale=True))
	    # datagen.flowは無限ループするため必要な枚数取得できたらループを抜ける
	    if (len(imgs) % max_img_num) == 0:
	        break
	show_imgs(imgs, row=4, col=4)
	