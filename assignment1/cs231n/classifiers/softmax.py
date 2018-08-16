import numpy as np
from random import shuffle
import math


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_class = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train):
        scores = X[i].dot(W)
        scores -= scores.max()    # for numeric stability
        correct_class_score = scores[y[i]]
        scores_sum = 0.0
        # for loss
        for j in range(num_class):
            scores_sum += math.exp(scores[j])
        # Li = -fyi + log(sigma(e^fj))
        loss += -correct_class_score + math.log(scores_sum)
        # for dW
        for j in range(num_class):
            dW[:, j] += (math.exp(scores[j]) /
                         scores_sum - (j == y[i])) * X[i]

    loss = loss / num_train + reg * np.sum(W * W)
    dW = dW / num_train + 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]

    # calculate scores with numeric stability
    scores = X.dot(W)
    scores_exp = np.exp(scores - scores.max(axis=1, keepdims=True))
    sum_scores_exp = np.sum(scores_exp, axis=1, keepdims=True)
    prob = scores_exp / sum_scores_exp

    # calculate loss
    loss = -np.log(prob[range(num_train), y]).sum()
    loss /= num_train
    loss += reg * np.sum(W * W)

    # calculate dW
    correct_W = np.zeros_like(prob)
    correct_W[range(num_train), y] = 1
    dW = X.T.dot(prob - correct_W)
    dW /= num_train
    dW += 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
