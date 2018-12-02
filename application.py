
import numpy as np
import os
import matplotlib.pyplot as plt
import pca
from scipy import linalg as LA

def compress_images(DATA, k):

    output = '.\output'
    if not os.path.exists(output):
        os.makedirs(output)

    #data is an array of tuples where item 1: is the file name item 2: is the image array
    images = []
    name = []
    for image_tuple in DATA:
        name.append(image_tuple[0])
        images.append(image_tuple[1])

    images = np.transpose(np.asarray(images))

    Z = pca.compute_Z(images)
    COV = pca.compute_covariance_matrix(Z)
    L, PCS = pca.find_pcs(COV)
    Z_star = pca.project_data(Z, PCS, L, k, 0)

    PCS = np.delete(PCS, range(k, PCS.shape[1]), axis=1)
    u_t = np.transpose(PCS)
    x_compressed = np.dot(Z_star, u_t)

    minimum = np.min(x_compressed)

    for value in x_compressed:
        value += abs(minimum)

    for column in range(0, images.shape[1]):
        image = np.reshape(x_compressed[:,column], (60, 48))
        plt.imsave( output + '\\' + name[column] + '_compressed.jpg', image, cmap='gray')

def load_data(input_dir):

    images = []
    for root, directories, files in os.walk(input_dir):
        for fileName in files:
            contents = plt.imread(root + fileName, 'pgm')
            flattened_image = np.asarray(contents.flatten(), dtype=float)
            images.append((fileName, flattened_image))

    return images
    #return np.transpose(np.asarray(images, dtype=float))


data = load_data('Data/Train/')
compress_images(data, 1000)
