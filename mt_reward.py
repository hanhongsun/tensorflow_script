import Image
import numpy as np
import sys
sys.path.append('/home/hanhong/Projects/python27/DeepLearningTutorials/code/')
from utils import tile_raster_images

def muscleTorqueScore(size, side, x):
    # size == side * side
    assert(size == x.shape[0])
    len = x.shape[1] # number of samples
    x = x * 2
    x -= 1
    # print x
    H = side/2 # half the side is the bone in center of the muscle 
    tqx = np.zeros((1,len))
    tqy = np.zeros((1,len))
    for i in range(-side + 1, side + 1, 2):
        for j in range(-side + 1, side + 1, 2):
            if i*i + j*j <= size:
                ind = side *(i + side -1)/2 + (j + side -1)/2
                tqx += x[ind, :]*i
                tqy += x[ind, :]*j
    # center of mass is the torque center == 4*r/(3*pi), mass of force = pi*r^2
    # raw score is 2 * 4/3 *r^3, as i, j ranged 2*d instead of d
    # size*side = 8*r^3, 
    score = 3 * np.sqrt(tqx*tqx + tqy*tqy)/(size*side)
    return score


if __name__ == "__main__":
    side_x = 100
    size_x = side_x * side_x

    # high score examples should be equal or close to 1
    x_test = np.zeros((size_x, 20))
    # print x_test
    x_test[0: size_x/2, 0] = 1
    x_test[size_x/2: size_x, 1] = 1
    logic_index = (np.array(range(0, size_x))/side_x) > (side_x - np.array(range(0, size_x))%side_x)
    x_test[logic_index, 2] = 1
    logic_index = (np.array(range(0, size_x))/side_x) < (side_x - np.array(range(0, size_x))%side_x)
    x_test[logic_index, 3] = 1
    logic_index = (np.array(range(0, size_x)) * 2 / side_x)%2 == 1
    x_test[logic_index, 4] = 1
    logic_index = (np.array(range(0, size_x)) * 2 / side_x)%2 == 0
    x_test[logic_index, 5] = 1
    logic_index = (np.array(range(0, size_x))/side_x) > (np.array(range(0, size_x))%side_x)
    x_test[logic_index, 6] = 1
    logic_index = (np.array(range(0, size_x))/side_x) < (np.array(range(0, size_x))%side_x)
    x_test[logic_index, 7] = 1
    
    # negtive examples, should keep as close to 0 as possible
    x_test[:, 8] = 1.0 * np.ones((size_x))
    x_test[0, 8] = 0
    # 9 is all zeros
    x_test[0:size_x:2, 10] = 1
    logic_index = (np.array(range(0, size_x))/side_x)%2 == 0
    x_test[logic_index, 11] = 1
    x_test[0:size_x:3, 12] = 1
    logic_index = (np.array(range(0, size_x))/side_x)%3 == 0
    x_test[logic_index, 13] = 1
    for i in range(-side_x + 1, side_x + 1, 2):
        for j in range(-side_x + 1, side_x + 1, 2):
            if i*i + j*j <= size_x:
                ind = side_x *(i + side_x -1)/2 + (j + side_x -1)/2
                x_test[ind, 14] = 1
    x_test[:, 15:21] = np.random.randint(2, size=(size_x, 5))
    # Do the test here
    print x_test
    print muscleTorqueScore(size_x, side_x, x_test)

    image = Image.fromarray(tile_raster_images(np.transpose(x_test),
                                           img_shape=(side_x, side_x),
                                           tile_shape=(2, 10),
                                           tile_spacing=(2, 2)))
    image.show()
