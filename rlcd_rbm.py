import tensorflow as tf
import Image
import numpy as np
import sys
sys.path.append('/home/hanhong/Projects/python27/DeepLearningTutorials/code/')
from utils import tile_raster_images
from tensorflow.python.ops import control_flow_ops

# size_x is the size of the visiable layer
# size_h is the size of the hidden layer
side_h = 20
size_x = 28*28
size_h = side_h * side_h
size_bt = 100 # batch size


