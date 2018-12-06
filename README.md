# Hopfield Network

This program is a **general** implementaion of [discrete Hopfield Network](http://en.wikipedia.org/wiki/Hopfield_network).

Discrete Hopfield Network can learn(memorize) patterns and remember(recover) the patterns when the network feeds those with noises.

## Getting Started

### Prerequisites

Prior to running this program, please install the following libraries.

-   `numpy`
-   `PIL`

### Example（Image Restoration）

#### Step1

Input a neat picture like this(yosukekatada's smile face).

![train](assets/yosuke.jpg)

#### Step2

Get the network to memorize the pattern, this program will automatically transform RGB Jpg into black-white picture.

#### Step3

After the network memorized it, put the picture with noise like this(yosukekatada's smile face with **sunglasses**) into the network.

![test](assets/yosuke_test.jpg)

#### Step4

The network can strip off the sunglasses, because the network ready remembers the former picture.

![recovery](assets/recovery_0.jpg)

## Basic Usage

1. Prepare data
1. Prerocess data
1. Train network
1. Test network
1. Show result

There are a example in [main.py](/main.py).

### Built With

1. `Hopfield.py`

    > General api for hopfield.

1. `utils.py`

    > Preprocess utils for program.

1. `main.py`

    > The enter to main program.

### Some Parameters

-   `theta`

    > The threshold of the neuron activation.

-   `epochs`

    > A parameter telling the steps of remembering the learned pictures. As the number of the steps increases, the remembered picture is more accurate.

-   `size`

    > The picture size in pixel. If you put a picture with different sizes, it will resize it.

-   `threshold`

    > The cutoff threshold to binarize 1 byte (from 0 to 255) brightness.

## Authors

| <img src="https://avatars3.githubusercontent.com/u/4463558?v=4" alt="yosukekatada" width="100px" height="100px"/> | <img src="https://avatars1.githubusercontent.com/u/25895405?v=4" alt="Zeroto521" width="100px" height="100px"/> |
| :---------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------: |
|                                  [yosukekatada](https://github.com/yosukekatada)                                  |                                    [Zeroto521](https://github.com/Zeroto521)                                    |

## License

MIT License. [@yosukekatada](https://github.com/yosukekatada), [@Zeroto521](https://github.com/Zeroto521)

## References

-   http://en.wikipedia.org/wiki/Hopfield_network
-   http://rishida.hatenablog.com/entry/2014/03/03/174331 (in Japanese)
