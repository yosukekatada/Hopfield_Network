# Hopfield Network

This Python code is just a simple implementaion of discrete [Hopfield Network](http://en.wikipedia.org/wiki/Hopfield_network).

Discrete Hopfield Network can learn(memorize) patterns and remember(recover) the patterns when the network feeds those with noises.

## Example (What the code do)

For example, you input a neat picture like this and get the network to memorize the pattern (My code automatically transform RGB Jpeg into black-white picture).

![IMAGE](train_pics/yosuke.jpg)

After the network memorized it, you put the picture with noise(sunglasses) like this into the network.

![IMAGE](test_pics/yosuke_test.jpg)

The network can strip off the sunglasses because the network remembers the former picture.

![IMAGE](after_1.jpeg)

## Instructions

### Input files

JPEG files like those in "train_pics".
Network learns those pics as correct pics.
If you want to add new pics, please put them in "train_pics" folder.

### Test files

The pictures with sunglasses should be in "test_pics" folder.

### Requirements

Prior to running this program, please install the following libraries.

-   `numpy`
-   `PIL`

### How to run the code

-   Theta is the threshold of the neuron activation.
-   epochs is a parameter telling the steps of remembering the learned pictures. As the number of the steps increases, the remembered picture is more accurate.
-   size is the picture size in pixel. If you put a pic with different sizes, the code resize it.
-   threshold is the cutoff threshold to binarize 1 byte (0 to 255) brightness.
-   current_path should be current working folder path (usual way is os.getcwd())

After you download all the files in this repository, please run "hopfield.py". Here is the main code.

```python
cwd_path = os.getcwd()

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
```

## Reference

-   http://en.wikipedia.org/wiki/Hopfield_network
-   http://rishida.hatenablog.com/entry/2014/03/03/174331 (in Japanese)
