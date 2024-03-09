# PyCNN: Image Processing with Cellular Neural Networks in Python

**Cellular Neural Networks (CNN)** [[wikipedia]](https://en.wikipedia.org/wiki/Cellular_neural_network) [[paper]](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7600) are a parallel computing paradigm that was first proposed in 1988. Cellular neural networks are similar to neural networks, with the difference that communication is allowed only between neighboring units. Image Processing is one of its [applications](https://en.wikipedia.org/wiki/Cellular_neural_network#Applications). CNN processors were designed to perform image processing; specifically, the original application of CNN processors was to perform real-time ultra-high frame-rate (>10,000 frame/s) processing unachievable by digital processors.

This python library is the implementation of CNN for the application of **Image Processing**.

**Note**: The library has been **cited** in the research published on [Using Python and Julia for Efficient Implementation of Natural Computing and Complexity Related Algorithms](http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=7168488), look for the reference #19 in the references section. I'm glad that this library could be of help to the community.

**Note**: Cellular neural network (CNN) must not be confused with completely different convolutional neural network (ConvNet).

![alt text](images/blockdiagram.png "CNN Architecture")

As shown in the above diagram, imagine a control system with a feedback loop. f(x) is the piece-wise linear sigmoid function. The control (template B) and the feedback (template A) templates (coefficients) are configurable and controls the output of the system. Significant research had been done in determining the templates for common image processing techniques, these templates are published in this [Template Library](http://cnn-technology.itk.ppke.hu/Template_library_v4.0alpha1.pdf).

## Dependencies

The library is supported for Python >= 2.7 and Python >= 3.3.

The python modules needed in order to use this library.
```
Pillow: 3.3.1
Scipy: 0.18.0
Numpy: 1.11.1 + mkl
```
Note: Scipy and Numpy can be installed on a Windows machines using binaries provided over [here](http://www.lfd.uci.edu/%7Egohlke/pythonlibs).

## Example

```sh
$ python3 main.py
```

**Input:**

![]()

**Output:**

![]()

## Usage

Import module

```python
from pycnn import PyCNN
import cv2
```

Initialize object

```python
cnn = PyCNN()
```

General image processing

```python
cnn.generalTemplates(name, inputImageLocation, outputImageLocation, tempA_A, tempB_B, 
                      initialCondition, Ib_b, t)
```

Corner detection

```python
cnn.cornerDetection(inputImageLocation, outputImageLocation)
```
