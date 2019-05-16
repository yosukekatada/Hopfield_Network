# This is the sample code of discrere hopfield network
import os
import random
import re

import numpy as np
from PIL import Image


def matrix2vector(x):
    """
    >>> import numpy as np
    >>> x = np.zeros(shape=(2, 2), dtype=np.float)
    >>> matrix2vector(x)
    array([0., 0., 0., 0.])
    """
    m = x.shape[0] * x.shape[1]
    v = np.zeros(m)

    c = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            v[c] = x[i, j]
            c += 1
    return v


# Create Weight matrix for a single image
def create_W(x):
    if x.ndim != 1:
        print "The input is not vector"
        return

    length = len(x)
    w = np.zeros([length, length])
    for i in range(length):
        for j in range(i, length):
            if i == j:
                w[i, j] = 0
            else:
                w[i, j] = x[i] * x[j]
                w[j, i] = w[i, j]
    return w


# Read Image file and convert it to Numpy array
def readImg2array(file, size, threshold=145):
    pilIN = Image.open(file).convert(mode="L")
    pilIN = pilIN.resize(size)
    # pilIN.thumbnail(size,Image.ANTIALIAS)
    imgArray = np.asarray(pilIN, dtype=np.uint8)
    x = np.zeros(imgArray.shape, dtype=np.float)
    x[imgArray > threshold] = 1
    x[x == 0] = -1
    return x


# Convert Numpy array to Image file like Jpeg
def array2img(data, outFile=None):
    # data is 1 or -1 matrix
    y = np.zeros(data.shape, dtype=np.uint8)
    y[data == 1] = 255
    y[data == -1] = 0
    img = Image.fromarray(y, mode="L")
    if outFile is not None:
        img.save(outFile)
    return img


# Update
def update(w, y_vec, theta=0.5, time=100):
    for s in range(time):
        m = len(y_vec)
        i = random.randint(0, m-1)
        u = np.dot(w[i][:], y_vec) - theta

        if u > 0:
            y_vec[i] = 1
        elif u < 0:
            y_vec[i] = -1

    return y_vec


# The following is training pipeline
# Initial setting
def hopfield(train_files, test_files, theta=0.5, time=1000,
             size=(100, 100), threshold=60, current_path=None):

    # read image and convert it to Numpy array
    print "Importing images and creating weight matrix...."

    # num_files is the number of files
    num_files = 0
    for path in train_files:
        print path
        x = readImg2array(file=path, size=size, threshold=threshold)
        x_vec = matrix2vector(x)
        print len(x_vec)
        if num_files == 0:
            w = create_W(x_vec)
            num_files = 1
        else:
            tmp_w = create_W(x_vec)
            w = w + tmp_w
            num_files += 1

    print "Weight matrix is done!!"

    # Import test data
    counter = 0
    for path in test_files:
        y = readImg2array(file=path, size=size, threshold=threshold)
        oshape = y.shape
        y_img = array2img(y)
        y_img.show()
        print "Imported test data"

        y_vec = matrix2vector(y)
        print "Updating..."
        y_vec_after = update(w=w, y_vec=y_vec, theta=theta, time=time)
        y_vec_after = y_vec_after.reshape(oshape)
        if current_path is None:
            after_img = array2img(y_vec_after, outFile=None)
            after_img.show()
        else:
            filename = "after_" + str(counter) + ".jpeg"
            outfile = os.path.join(current_path, filename)
            array2img(y_vec_after, outFile=outfile)
        counter += 1


def get_image_files(target_dir):
    for filename in os.listdir(target_dir):
        if re.match(r'[0-9a-zA-Z-_]*.jp[e]*g', filename):
            yield os.path.join(target_dir, filename)


def main():
    # First, you can create a list of input file path
    current_path = os.getcwd()
    train_paths = get_image_files(os.path.join(current_path, 'train_pics'))

    # Second, you can create a list of sungallses file path
    test_paths = get_image_files(os.path.join(current_path, 'test_pics'))

    # Hopfield network starts!
    hopfield(
        train_files=train_paths, test_files=test_paths,
        theta=0.5, time=20000, size=(100, 100), threshold=60,
        current_path=current_path)


if __name__ == '__main__':
    main()
