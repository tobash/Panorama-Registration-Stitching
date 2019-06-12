import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.ndimage.filters import convolve
from scipy import signal
from scipy.ndimage import convolve as con


GRASCALE_REPRE = 1
RGB_REPRE = 2
GRASCALE_SHAPE = 2
RGB_SHAPE = 3
COLOR_LEVEL = 256
CONV_MAT = np.array([1, 1])


def read_image(filename, representation):

    '''
    A function that converts the image to a desired representation, and with
    intesities normalized to the range of [0,1]
    :param filename: the filename of an image on disk, could be grayscale or
    RGB
    :param representation: representation code, either 1 or 2 defining whether
    the output should be a grayscale image (1) or an RGB image (2)
    :return: an image in the desired representation.
    '''

    im = imread(filename)
    if representation == GRASCALE_REPRE:
        im = rgb2gray(im)
    im_float = im.astype(np.float64)
    im_float /= (255)
    return im_float



def build_gaussian_pyramid(im, max_levels, filter_size):

    '''
    Function that construct a Gaussian pyramid of a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the Gaussian filter (an odd scalar that
    represents a squared filter) to be used in constructing the pyramid filter
    :return:
    filter_vec - row vector of shape (1, filter_size) used for the pyramid
    construction
    pyr - a standard python array with maximum length of max_levels, where each
    element of the array is a grayscale image.
    '''

    filter_vec = np.array([1, 1])
    pyr = [im]
    im_n = im
    for i in range(filter_size-2):
        filter_vec = np.convolve(filter_vec, CONV_MAT)
    filter_vec = (1 / np.sum(filter_vec)) * filter_vec[None,:]


    while (len(pyr) < max_levels and np.shape(pyr[-1])[0] / 2 >= 16 and
                       np.shape(pyr[-1])[1] / 2 >= 16):
        im_n = convolve(im_n, filter_vec, mode='nearest')
        im_n = (convolve(im_n.transpose(), filter_vec, mode='nearest')).transpose()
        im_n = im_n[::2, ::2]
        pyr.append(im_n)

    return [pyr, filter_vec]


def blur_spatial (im, kernel_size):

    '''
    function that performs image blurring using 2D convolution between the
    image and a gaussian kernel.
    :param im: image to be blurred (grayscale float64 image).
    :param kernel_size: size of the gaussian kernel in each dimension
    (an odd integer).
    :return: blurry image (grayscale float64 image)
    '''

    kernel_mat = np.array([1,1],np.float64)
    conv_mat = np.array([1,1])

    for i in range(kernel_size):
        kernel_mat = np.convolve(kernel_mat,conv_mat)

    kernel_mat = (np.expand_dims(kernel_mat,axis=0)).transpose()
    conv_mat = np.expand_dims(conv_mat,axis=0)

    for i in range(kernel_size):
        kernel_mat = signal.convolve2d(kernel_mat,conv_mat)

    kernel_mat *= 1/np.sum(kernel_mat)
    return convolve(im,kernel_mat)
