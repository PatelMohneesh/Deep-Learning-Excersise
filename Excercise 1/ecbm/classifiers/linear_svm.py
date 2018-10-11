import numpy as np

def svm_loss_naive(W, X, y, reg):
    """
    Multi-class Linear SVM loss function, naive implementation (with loops).
    
    In default, delta is 1 and there is no penalty term wst delta in objective function.

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: a numpy array of shape (D, C) containing weights.
    - X: a numpy array of shape (N, D) containing N samples.
    - y: a numpy array of shape (N,) containing training labels; y[i] = c means
         that X[i] has label c, where 0 <= c < C.
    - reg: (float) L2 regularization strength

    Returns:
    - loss: a float scalar
    - gradient: wrt weights W, an array of same shape as W
    """
    dW = np.zeros(W.shape).astype('float') # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:,j] += X[i]
                dW[:,y[i]] -= X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg*2*W

    return loss, dW

def svm_loss_vectorized(W, X, y, reg):
    """
    Linear SVM loss function, vectorized implementation.
    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape).astype('float') # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    num_train = X.shape[0]
    scores = X.dot(W)
    mask = np.arange(num_train)
    correct_class_scores = scores[mask, y][:, np.newaxis]
    margins = np.maximum(0, scores - correct_class_scores + 1)
    margins[mask, y] = 0
    loss = np.sum(margins)
    loss /= num_train
    loss += reg * np.sum(W**2)
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    margin_counts = np.zeros(margins.shape)
    margin_counts[margins > 0] = 1
    margin_counts[mask, y] = -np.sum(margins > 0, axis=1)
    dW = X.T.dot(margin_counts)
    dW /= num_train
    dW += reg *2* W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
    
