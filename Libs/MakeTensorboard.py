# -*- coding: utf-8 -*-
"""
tensorboardで学習率/正答率のログを記録していく。
set_dir_nameでセットしたディレクトリにログファイルが生成される。
TensorBoardでログファイルを読み込むことで簡単にグラフ化してモデルの分析ができる。
使い方:
https://www.tensorflow.org/tensorboard/get_started?hl=ja
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from time import gmtime, strftime
from keras.callbacks import TensorBoard
import os



def make_tensorboard(set_dir_name=''):
    tictoc = strftime("%a_%d_%b_%Y_%H_%M_%S", gmtime())
    directory_name = tictoc
    log_dir = set_dir_name + '_' + directory_name
    os.mkdir(log_dir)
    tensorboard = TensorBoard(log_dir=log_dir, write_graph=True, )
    return tensorboard
