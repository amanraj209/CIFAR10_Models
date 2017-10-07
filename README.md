# CIFAR-10 MODELS

This project tests the accuracies of several Machine Learning models on the CIFAR-10 dataset.

## Models tested on the Dataset

* Logistic Regression (Binary Classification)
* One-vs-All Logistic Regression (Multiclass Classification)
* SVM Classification
* Softmax Classification

## CIFAR-10 Dataset
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class. 

Download the CIFAR-10 Dataset from [here](http://www.cs.toronto.edu/~kriz/cifar.html).

After downloading the dataset, extract it and place all the data_batch files under datasets folder

## Folder Structure

```
CIFAR-10/
    README.md
    setup.py
    logistic_regression.py
    one_vs_all_LR.py
    datasets/
    algorithms/
        data_utils.py
        gradient_check.py
        classifiers/
            linear_classifier.py
            loss_grad_logistic.py
            loss_grad_softmax.py
```

**NOTE:** Use python 3 and install all the required libraries using `pip install` command.

## Contributions

If there is any error in the source code, send me pull request and contribute to this project.