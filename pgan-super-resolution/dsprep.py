import numpy as np
import tensorflow as tf
from matplotlib import pyplot
from skimage.transform import resize
from tqdm import tqdm


def get_numpy_array(filename):
    print("Loading samples from file '{}'".format(filename))
    # load dataset
    data = np.load(filename)
    # extract numpy array
    return data['arr_0']


# load dataset
def normalized(dataset):
    # convert from ints to floats
    X = dataset.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    print("\t* {} images loaded and normalized".format(len(X)))
    print("\t* Image shape: {}".format(X.shape[1:4]))
    return X


# scale images to preferred size
def scale(images, new_shape):
    images_list = list()
    print('Scaling images from {} to {}'.format(images[0].shape, new_shape))
    #for image in tqdm(images[0:10000]):
        # resize with nearest neighbor interpolation
    #    new_image = resize(image, new_shape, 0)
        # store
    #    images_list.append(new_image)

    #print('Images scaled')
    #return np.asarray(images_list)
    return tf.image.resize(images, new_shape[0:2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR).numpy()


def plot_faces(faces, n):
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(faces[i].astype('uint8'))
    pyplot.show()
