import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
      Softmax loss function, naive implementation (with loops)

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
      - reg: (float) regularization strength

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wst W, an array of same shape as W
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
    num_train = X.shape[0]
    num_classes = W.shape[1]
    for i in range(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores)
        exp_sum = np.sum(np.exp(scores))
        softmax = np.exp(scores[y[i]]) / exp_sum
        loss -= np.log(softmax)
        scores_p =np.exp(scores)/exp_sum
        for j in range(num_classes):
            if j == y[i]:
                dscore = scores_p[j] - 1
            else:
                dscore = scores_p[j]
            dW[:,j] += dscore * X[i]        
    loss /= num_train
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += reg *2* W
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
    scores = np.dot(X, W)
    scores -= np.max(scores, axis=1, keepdims=True)
    probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    smooth_factor = 1e-14
    N = X.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y] + smooth_factor)) / N
    dscores = probs.copy()
    dscores[np.arange(N), y] -= 1
    dscores /= N
    dW = np.dot(X.T, dscores)
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
