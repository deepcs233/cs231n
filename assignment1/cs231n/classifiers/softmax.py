import numpy as np
from random import shuffle

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

  train_num = X.shape[0]
  s = X.dot(W)
  f_max = np.reshape(np.max(s,axis=1),(train_num,1))

  s = np.exp(s-f_max)
  sums = np.sum(s,axis=1).reshape(500,1)
  s = s/sums

  class_num = W.shape[1]
  for i in range(train_num):
    loss -= np.log(s[i,y[i]])

  loss /= train_num
  loss += 0.5 * reg * np.sum(W * W)

  for i in range(train_num):
    for j in range(class_num):
      if j == y[i]:
        aa = X[i].dot(W[:,j]) / sum(X[i].dot(W))
        aa = s[i,j]
        dW[:, j] += aa * X[i]
        dW[:, j] -= X[i]
      else:
        aa = X[i].dot(W[:,j]) / sum(X[i].dot(W))

        aa = s[i,j]
        dW[:, j] += aa * X[i]

  dW /= train_num
  dW += 2 * reg * W
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


  train_num = X.shape[0]
  s = X.dot(W)
  f_max = np.reshape(np.max(s,axis=1),(train_num,1))

  s = np.exp(s-f_max)
  sums = np.sum(s,axis=1).reshape(train_num,1)
  s = s/sums

  class_num = W.shape[1]
  loss -= np.sum(np.log(s[range(train_num),y]))
  loss /= train_num
  loss += 0.5 * reg * np.sum(W * W)


  s[range(train_num),y] -= 1
  dW += X.T.dot(s)
  dW /= train_num
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

