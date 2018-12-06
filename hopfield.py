import os
import random

import numpy as np

from PIL import Image


def readImg2array(file, size, threshold=145):
    """Read Image file and convert it to Numpy array.

    Arguments:
        file {str} -- image local file path
        size {tuple} -- resize image size

    Keyword Arguments:
        threshold {int} -- binary threshold (default: {145})

    Returns:
        np.ndarray -- image matrix
    """

    # read image file and convert it to gray
    img = Image.open(file).convert(mode="L")
    img = img.resize(size)  # resize image

    # binarize image array
    imgArray = np.asarray(img)
    mat = np.zeros(img.size)

    mat[imgArray > threshold] = 1
    mat[imgArray <= threshold] = -1

    return mat


def array2img(data, outFile=None):
    """Convert Numpy array to Image file like '*.jpg'.

    Arguments:
        data {np.ndarray} -- 1 or -1 matrix

    Keyword Arguments:
        outFile {str} -- output file path (default: {None})

    Returns:
        Image.image -- imgage
    """

    mat = np.zeros(data.shape, dtype=np.uint8)
    mat[data == 1] = 255
    mat[data == -1] = 0
    img = Image.fromarray(mat)
    if outFile is not None:
        img.save(outFile)

    return img


def create_W(x):
    """Create Weight matrix for a single image

    Arguments:
        x {np.ndarray} -- vector

    Returns:
        np.ndarray -- the weight of input(x)
    """
    # TODO: spend too time to do loop, anyway.
    assert len(x.shape) == 1, "The input is not vector"

    length = len(x)
    w = np.zeros((length, length))
    for i in range(length):
        for j in range(i, length):
            if i != j:
                w[i, j] = x[i]*x[j]
                w[j, i] = w[i, j]
    return w


def update(w, y_vec, theta=0.5, epochs=100):
    length = len(y_vec)
    for _ in range(epochs):
        ind = random.randint(0, length-1)
        u = np.dot(w[ind], y_vec) - theta
        if u > 0:
            y_vec[ind] = 1
        elif u < 0:
            y_vec[ind] = -1

    return y_vec


def hopfield(train_files, test_files, theta=0.5, time=1000, size=(100, 100), threshold=60, cwd_path=None):
    """Training pipeline.

    Arguments:
        train_files {list} -- train data path list
        test_files {list} -- test data path list

    Keyword Arguments:
        theta {float} -- [description] (default: {0.5})
        time {int} -- [description] (default: {1000})
        size {tuple} -- [description] (default: {(100, 100)})
        threshold {int} -- [description] (default: {60})
        cwd_path {[type]} -- [description] (default: {None})
    """

    print("Loading images and creating weight matrix....")

    # TODO: split read images and create weight
    counter = 0  # num_files is the number of files
    for path in train_files:
        x = readImg2array(file=path, size=size, threshold=threshold)
        x_vec = x.flatten()
        if counter == 0:
            w = create_W(x_vec)
        else:
            tmp_w = create_W(x_vec)
            w = w + tmp_w
        counter += 1

    print("Weight matrix is done!")

    print("Imported test data.")
    print("Updating...")
    # TODO: split read images and create weight
    counter = 0
    for path in test_files:
        y = readImg2array(file=path, size=size, threshold=threshold)
        y_vec = y.flatten()
        y_vec_after = update(w=w, y_vec=y_vec, theta=theta, epochs=time)
        y_vec_after = y_vec_after.reshape(y.shape)
        if cwd_path is not None:
            outfile = os.path.join(cwd_path, 'recovery_'+str(counter)+".jpg")
            array2img(y_vec_after, outFile=outfile)
        counter += 1


if __name__ == "__main__":
    cwd_path = os.getcwd()

    # TODO: build hopfield class

    # First, you can create a list of input file path
    train_path = os.path.join(cwd_path, 'train_pics')
    train_paths = [os.path.join(train_path, p) for p in os.listdir(train_path)]

    # Second, you can create a list of sungallses file path
    test_path = os.path.join(cwd_path, 'test_pics')
    test_paths = [os.path.join(test_path, p) for p in os.listdir(test_path)]

    # Hopfield network starts!
    hopfield(
        train_files=train_paths,
        test_files=test_paths,
        theta=0.5, time=20000, size=(100, 100),
        threshold=60, cwd_path=cwd_path
    )
