import os

import Hopfield
import utils

# paramters
cwd_path = os.getcwd()
theta = 0.5
threshold = 60
epochs = 20000
size = (100, 100)


# create a list of train data.
train_path = os.path.join(cwd_path, 'data/train_pics')
train_data = utils.preprocess(train_path, size, threshold)

# create a list of tese data, e.g. yosukekatada 's sunglasses picture.
test_path = os.path.join(cwd_path, 'data/test_pics')
test_data = utils.preprocess(test_path, size, threshold)


# build network
network = Hopfield.Hopfield(isload=True)
network.train(train_data)  # start training
recovery = network.predict(test_data, epochs=epochs)  # start testing

# recovery image
for counter, img in enumerate(recovery):
    img_after = img.reshape(size)
    filename = "recovery_{}.jpg".format(counter)
    outfile = os.path.join(cwd_path, filename)
    utils.array2img(img_after, outFile=outfile)
