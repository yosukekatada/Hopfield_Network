import random

import numpy as np


class Hopfield(object):

    def create_W(self, x):
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

    def update(self, y, theta=0.5, epochs=100):
        """Update test sample.

        Arguments:
            y {np.ndarray} -- vector

        Keyword Arguments:
            theta {float} -- the threshold of the neuron activation(default: {0.5})
            epochs {int} -- the max iteration of loop(default: {100})

        Returns:
            np.ndarray -- recoveried sample
        """

        length = len(y)
        for _ in range(epochs):
            ind = random.randint(0, length-1)
            u = np.dot(self.w[ind], y) - theta
            if u > 0:
                y[ind] = 1
            elif u < 0:
                y[ind] = -1

        return y

    def train(self, data):
        """Training pipeline.

        Arguments:
            data {list} -- each sample is vector
        """

        print("Loading images and creating weight matrix....")
        counter = 0  # the number of data
        for x in data:
            if counter == 0:
                self.w = self.create_W(x)
            else:
                tmp_w = self.create_W(x)
                self.w = self.w + tmp_w
            counter += 1

        print("Weight matrix is done!")

    def predict(self, data, epochs=1000, theta=0.5):
        """Predicting pipline.

        Arguments:
            data {list} -- each sample is vector

        Keyword Arguments:
            epochs {int} -- the max iteration of loop (default: {1000})
            theta {float} -- the threshold of the neuron activation (default: {0.5})

        Returns:
            list -- recoveried by hopfield data 
        """

        counter = 0
        recovery = []
        for y in data:
            counter += 1
            recovery.append(self.update(y=y, theta=theta, epochs=epochs))

            print("The {}th sample is updating...".format(counter))

        return recovery
