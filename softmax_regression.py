import time
import numpy as np
from ggplot import *
import matplotlib.pyplot as plt
from algorithms.data_utils import load_CIFAR10
from algorithms.classifiers.loss_grad_softmax import *
from algorithms.gradient_check import gradient_check_sparse
from algorithms.classifiers.linear_classifier import Softmax

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def get_CIFAR10_data(num_training = 49000, num_val = 1000, num_test = 10000, show_sample = True):
    """
    Load the CIFAR-10 dataset, and divide the sample into training set, validation set and test set
    """
    cifar10_dir = 'datasets/'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    
    # subsample the data for validation set
    mask = range(num_training, num_training + num_val)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]
    return X_train, y_train, X_val, y_val, X_test, y_test

def subset_classes_data(classes):
    # Subset 'plane' and 'car' classes to perform logistic regression
    idxs = np.logical_or(y_train_raw == 0, y_train_raw == 1)
    X_train = X_train_raw[idxs, :]
    y_train = y_train_raw[idxs]
    # validation set
    idxs = np.logical_or(y_val_raw == 0, y_val_raw == 1)
    X_val = X_val_raw[idxs, :]
    y_val = y_val_raw[idxs]
    # test set
    idxs = np.logical_or(y_test_raw == 0, y_test_raw == 1)
    X_test = X_test_raw[idxs, :]
    y_test = y_test_raw[idxs]
    return X_train, y_train, X_val, y_val, X_test, y_test

def visualize_sample(X_train, y_train, classes, samples_per_class = 7):
    """visualize some samples in the training datasets """
    num_classes = len(classes)
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y) # get all the indexes of cls
        idxs = np.random.choice(idxs, samples_per_class, replace = False)
        for i, idx in enumerate(idxs): # plot the image one by one
            plt_idx = i * num_classes + y + 1 # i*num_classes and y+1 determine the row and column respectively
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()

def preprocessing_CIFAR10_data(X_train, y_train, X_val, y_val, X_test, y_test):
    # Preprocessing: reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1)) # [49000, 3072]
    X_val = np.reshape(X_val, (X_val.shape[0], -1)) # [49000, 3072]
    X_test = np.reshape(X_test, (X_test.shape[0], -1)) # [10000, 3072]
    
    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis = 0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    
    # Add bias dimension and transform into columns
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))]).T
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))]).T
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))]).T
    return X_train, y_train, X_val, y_val, X_test, y_test



# Invoke the above functions to get our data
X_train_raw, y_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw = get_CIFAR10_data()
visualize_sample(X_train_raw, y_test_raw, classes)
X_train, y_train, X_val, y_val, X_test, y_test = preprocessing_CIFAR10_data(X_train_raw, y_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw)



# As a sanity check, we print out th size of the training and test data dimenstion
print ('Train data shape : ', X_train.shape)
print ('Train labels shape : ', y_train.shape)
print ('Validation data shape : ', X_val.shape)
print ('Validation labels shape : ', y_val.shape)
print ('Test data shape : ', X_test.shape)
print ('Test labels shape : ', y_test.shape)

print ('\n\n\n')


# Softmax Regression Classifier
# generate a rand weights W 
W = np.random.randn(10, X_train.shape[0]) * 0.001
tic = time.time()
loss_naive, grad_naive = loss_grad_softmax_naive(W, X_train, y_train, 0)
toc = time.time()
print ('Naive loss : %f and gradient computed in %fs' % (loss_naive, (toc - tic)))

tic = time.time()
loss_vec, grad_vec = loss_grad_softmax_vectorized(W, X_train, y_train, 0)
toc = time.time()
print ('Vectorized loss : %f and gradient computed in %fs' % (loss_vec, (toc - tic)))



# Compare the gradient, because the gradient is a vector, we canuse the Frobenius norm to compare them
# the Frobenius norm of two matrices is the square root of the squared sum of differences of all elements
diff = np.linalg.norm(grad_naive - grad_vec, ord = 'fro')
# Randomly choose some gradient to check
idxs = np.random.choice(X_train.shape[0], 10, replace = False)
print (idxs)
print (grad_naive[0, idxs])
print (grad_vec[0, idxs])
print ('Gradient difference between naive and vectorized version is: %f' % diff)

print ('\n\n\n')


# Check gradient using numerical gradient along several randomly chosen dimension
f = lambda w : loss_grad_softmax_vectorized(w, X_train, y_train, 0)[0]
grad_numerical = gradient_check_sparse(f, W, grad_vec, 10)

print ('\n\n\n')


# Training softmax regression classifier using SGD and BGD
# using BGD algorithm
"""
softmax_bgd = Softmax()
tic = time.time()
losses_bgd = softmax_bgd.train(X_train, y_train, method = 'bgd', batch_size = 200,
                                learning_rate = 1e-6, reg = 1e2, num_iters = 1000,
                                verbose = True, vectorized = True)
toc = time.time()
print ('Traning time for BGD with vectorized version : %f \n' % (toc - tic))

# Compute the accuracy of training data and validation data using Logistic.predict function
y_train_pred_bgd = softmax_bgd.predict(X_train)[0]
print ('Training accuracy : %f' % (np.mean(y_train == y_train_pred_bgd)))
y_val_pred_bgd = softmax_bgd.predict(X_val)[0]
print ('Validation accuracy : %f' % (np.mean(y_val == y_val_pred_bgd)))

print ('\n')
"""

# using SGD algorithm
softmax_sgd = Softmax()
tic = time.time()
losses_sgd = softmax_sgd.train(X_train, y_train, method = 'sgd', batch_size = 200,
                                learning_rate = 1e-6, reg = 1e5, num_iters = 1000,
                                verbose = True, vectorized = True)
toc = time.time()
print ('Traning time for SGD with vectorized version : %f \n' % (toc - tic))

# Compute the accuracy of training data and validation data using Logistic.predict function
y_train_pred_sgd = softmax_sgd.predict(X_train)[0]
print ('Training accuracy : %f' % (np.mean(y_train == y_train_pred_sgd)))
y_val_pred_sgd = softmax_sgd.predict(X_val)[0]
print ('Validation accuracy : %f' % (np.mean(y_val == y_val_pred_sgd)))

# A useful degugging strategy is to plot the loss 
# as a function of iteration number:
# qplot(range(len(losses_bgd)), losses_bgd) + labs(x = 'Iteration Number', y = 'BGD Loss Value')
qplot(range(len(losses_sgd)), losses_sgd) + labs(x = 'Iteration Number', y = 'SGD Loss Value')


# Using validation set to tuen hyperparameters, i.e., learning rate and regularization strength
learning_rates = [1e-5, 1e-8]
regularization_strengths = [10e2, 10e4]

# Result is a dictionary mapping tuples of the form (learning_rate, regularization_strength) 
# to tuples of the form (training_accuracy, validation_accuracy). The accuracy is simply the fraction
# of data points that are correctly classified.
results = {}
best_val = -1
best_softmax = None

# Choose the best hyperparameters by tuning on the validation set
i = 0
interval = 5

for learning_rate in np.linspace(learning_rates[0], learning_rates[1], num = interval):
    i += 1
    print ('The current iteration is %d/%d' % (i, interval))
    for reg in np.linspace(regularization_strengths[0], regularization_strengths[1], num = interval):
        softmax = Softmax()
        softmax.train(X_train, y_train, method = 'sgd', batch_size = 200, learning_rate = learning_rate,
                       reg = reg, num_iters = 1000, verbose = False, vectorized = True)
        y_train_pred = softmax.predict(X_train)[0]
        y_val_pred = softmax.predict(X_val)[0]
        train_accuracy = np.mean(y_train == y_train_pred)
        val_accuracy = np.mean(y_val == y_val_pred)
        results[(learning_rate, reg)] = (train_accuracy, val_accuracy)
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_softmax = softmax
        else:
            pass

# Print out the results
for learning_rate, reg in sorted(results):
    train_accuracy, val_accuracy = results[(learning_rate, reg)]
    print ('learning rate %e and regularization %e, \n \
    the training accuracy is: %f and validation accuracy is: %f.\n' % (learning_rate, reg, train_accuracy, val_accuracy))
    

# Test best softmax classifier on test datasets
y_test_predict_result = best_softmax.predict(X_test)
y_test_predict = y_test_predict_result[0]
test_accuracy = np.mean(y_test_predict == y_test)
print ('The test accuracy is: %f' % test_accuracy)
