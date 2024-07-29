# ML based number recognizer

## Numpy
In main.py I have implemented a neural network with 2 hidden layers using only numpy. After optimizing the hyperparameters I have reached 94% accuracy on the testing data. With a shorter learning time currently it is around 90%.
In the future I am planning to implplement crossentropy loss and softmax in order to increase  the accuracy of the model.

## PyTorch
In intorch.py I have implemented the same neural network, but using softmax and crossentropy loss. Using cuda and I have achieved 97% accuracy much faster.

## Window
In window.py I have made a window in pygame, through which the user can input their own handdrawn numbers. It uses the model seen in main.py and works by starting the program.

![UI](https://github.com/Grabowskiy/ML-based-number-recognizer/blob/master/window.png)
