# -*- coding: utf-8 -*-
"""
@author: oxygen0605
"""


from functools import reduce
from keras import backend
from keras.layers import (Activation, Add, GlobalAveragePooling2D,BatchNormalization, Conv2D, Dense, Flatten, Input, MaxPooling2D)
from keras.models import Model
from keras.regularizers import l2

def compose(*funcs):
	'''
		複数の層を結合する。
	'''
	if funcs:
		return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), funcs)
	else:
		raise ValueError('Composition of empty sequence not supported.')
		
def ResNetConv2D(*args, **kwargs):
	'''
		conv を作成する。
	'''
	conv_kwargs = {
			'strides': (1, 1),
			'padding': 'same',
			'kernel_initializer': 'he_normal',
			'kernel_regularizer': l2(1.e-4)
	}
	
	# update:辞書の要素を更新
	conv_kwargs.update(kwargs) 
	
	return Conv2D(*args, **conv_kwargs)
	
def bn_relu_conv(*args, **kwargs):
	'''
		batch mormalization -> ReLU -> conv を作成する。
	'''
	return compose(
		BatchNormalization(),
		Activation('relu'),
		ResNetConv2D(*args, **kwargs))

def shortcut(x, residual):
	'''
		shortcut connection を作成する。
	'''
	x_shape = backend.int_shape(x)
	residual_shape = backend.int_shape(residual)
	
	if x_shape == residual_shape:
		# x と residual の形状が同じ場合、なにもしない。
		shortcut = x
	else:
		# x と residual の形状が異なる場合、線形変換を行い、形状を一致させる。
		stride_w = int(round(x_shape[1] / residual_shape[1]))
		stride_h = int(round(x_shape[2] / residual_shape[2]))
		
		shortcut = Conv2D(filters=residual_shape[3],
					 kernel_size=(1, 1),
					 strides=(stride_w, stride_h),
					 kernel_initializer='he_normal',
					 kernel_regularizer=l2(1.e-4))(x)
	
	return Add()([shortcut, residual])

# ResNet-18, ResNet-34用
def basic_block(filters, first_strides, is_first_block_of_first_layer):
	'''bulding block を作成する。
	
        Arguments:
            filters: フィルター数
            first_strides: 最初の畳み込みのストライド
            is_first_block_of_first_layer: max pooling 直後の residual block かどうか
	'''
	def f(x):
		if is_first_block_of_first_layer:
			# conv1 で batch normalization -> ReLU はすでに適用済みなので、
			# max pooling の直後の residual block は畳み込みから始める。
			conv1 = ResNetConv2D(filters=filters, kernel_size=(3, 3))(x)
		else:
			conv1 = bn_relu_conv(filters=filters, kernel_size=(3, 3),
						strides=first_strides)(x)
			
		conv2 = bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
		
		return shortcut(x, conv2)

	return f

# ResNet-50, ResNet-101, ResNet-152用
def bottleneck_block(filters, first_strides, is_first_block_of_first_layer):
	'''bottleneck bulding block を作成する。
	
	        Arguments:
	            filters: フィルター数
	            first_strides: 最初の畳み込みのストライド
	            is_first_block_of_first_layer: max pooling 直後の residual block かどうか
	'''

	def f(x):
		if is_first_block_of_first_layer:
			# conv1 で batch normalization -> ReLU はすでに適用済みなので、
			# max pooling の直後の residual block は畳み込みから始める。
			conv1 = ResNetConv2D(filters=filters, kernel_size=(3, 3))(x)
		else:
			conv1 = bn_relu_conv(filters=filters, kernel_size=(1, 1),
						strides=first_strides)(x)
		
		conv2 = bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
		conv3 = bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv2)
		
		return shortcut(x, conv3)

	return f

# 複数の residual block を反復させるヘルパー関数
def residual_blocks(block_function, filters, repetitions, is_first_layer):
	'''residual block を反復する構造を作成する。

        Arguments:
            block_function: residual block を作成する関数
            filters: フィルター数
            repetitions: residual block を何個繰り返すか。
            is_first_layer: max pooling 直後かどうか
	'''
	def f(x):
		for i in range(repetitions):
			# conv3_x, conv4_x, conv5_x の最初の畳み込みは、
			# プーリング目的の畳み込みなので、strides を (2, 2) にする。
			# ただし、conv2_x の最初の畳み込みは直前の max pooling 層でプーリングしているので
			# strides を (1, 1) にする。
			first_strides = (2, 2) if i == 0 and not is_first_layer else (1, 1)
			
			x = block_function(filters=filters, first_strides=first_strides,
					  is_first_block_of_first_layer=(i == 0 and is_first_layer))(x)
		
		return x
	
	return f

class ResnetBuilder():
	@staticmethod
	def build(input_shape, num_outputs, block_type, repetitions):
		'''ResNet モデルを作成する Factory クラス
        Arguments:
            input_shape: 入力の形状
            num_outputs: ネットワークの出力数
            block_type : residual block の種類 ('basic' or 'bottleneck')
            repetitions: 同じ residual block を何個反復させるか
		'''
		# block_type に応じて、residual block を生成する関数を選択する。
		if block_type == 'basic':
			block_fn = basic_block
		elif block_type == 'bottleneck':
			block_fn = bottleneck_block

		# モデルを作成する。
		##############################################
		input = Input(shape=input_shape)

		# conv1 (batch normalization -> ReLU -> conv)
		conv1 = compose(ResNetConv2D(filters=64, kernel_size=(7, 7), strides=(2, 2)),
						BatchNormalization(),
						Activation('relu'))(input)

		# pool
		pool1 = MaxPooling2D(
				pool_size=(3, 3), strides=(2, 2), padding='same')(conv1)
		
		# conv2_x, conv3_x, conv4_x, conv5_x
		block = pool1
		filters = 64
		for i, r in enumerate(repetitions):
			block = residual_blocks(block_fn, filters=filters, repetitions=r,
									is_first_layer=(i == 0))(block)
			filters *= 2

		# batch normalization -> ReLU
		block = compose(BatchNormalization(),
						Activation('relu'))(block)

		# global average pooling
		pool2 = GlobalAveragePooling2D()(block)

		# dense
		fc1 = Dense(units=num_outputs,
					kernel_initializer='he_normal',
					activation='softmax')(pool2)

		return Model(inputs=input, outputs=fc1)

	@staticmethod
	def resnet_18(input_shape, num_outputs):
		return ResnetBuilder.build(
			input_shape, num_outputs, 'basic', [2, 2, 2, 2])

	@staticmethod
	def resnet_34(input_shape, num_outputs):
		return ResnetBuilder.build(
			input_shape, num_outputs, 'basic', [3, 4, 6, 3])

	@staticmethod
	def resnet_50(input_shape, num_outputs):
		return ResnetBuilder.build(
			input_shape, num_outputs, 'bottleneck', [3, 4, 6, 3])

	@staticmethod
	def resnet_101(input_shape, num_outputs):
		return ResnetBuilder.build(
			input_shape, num_outputs, 'bottleneck', [3, 4, 23, 3])

	@staticmethod
	def resnet_152(input_shape, num_outputs):
		return ResnetBuilder.build(
			input_shape, num_outputs, 'bottleneck', [3, 8, 36, 3])


if __name__ == '__main__':

	input_shape = (32,32, 3)  # モデルの入力サイズ
	num_classes = 10  # クラス数

	# モデルを作成する。
	model = ResnetBuilder.resnet_34(input_shape, num_classes)
	#ResNetConv2D([1,2],1,3,{53,1})
	#print('finish')