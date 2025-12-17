import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
import pickle


np.random.seed(20)

N = 1000 # number of points per class
d0 = 2 # dimensionality
C = 3 # number of classes
X = np.zeros((N*C, d0)) # data matrix (each row = single example)
y = np.zeros(N*C, dtype='uint8') # class labels

for j in range(C):
    ix = range(N*j,N*(j+1))
    r = np.linspace(0.0,1,N) # radius
    t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j

# # lets visualize the data:
# plt.scatter(X[:N, 0], X[:N, 1], c=y[:N], s=40, cmap=plt.cm.Spectral)

# plt.plot(X[:N, 0], X[:N, 1], 'ws',  markersize = 6, mec = 'k');
# plt.plot(X[N:2*N, 0], X[N:2*N, 1], 'w^', markersize = 6, mec = 'k');
# plt.plot(X[2*N:, 0], X[2*N:, 1], 'ko', markersize = 6, mec = 'k');
# # plt.axis('off')
# plt.xlim([-1.5, 1.5])
# plt.ylim([-1.5, 1.5])
# cur_axes = plt.gca()
# cur_axes.axes.get_xaxis().set_ticks([])
# cur_axes.axes.get_yaxis().set_ticks([])
# filename = 'EX.pdf'
# with PdfPages(filename) as pdf:
#     pdf.savefig(bbox_inches='tight')
# # plt.savefig('EX.png', bbox_inches='tight', dpi = 600)
# plt.show()







"""
X: 2d numpy array of shape (N, d0), d0=2, C=3
y: a vector of size N representing class labels
W1: d0 x d1
B1, Z1 and A1: N x d1

W2: d1 x C
B2, Z2 and A2 (Y_hat): N x C

"""


def softmax_stable(Z):
    """
    Compute softmax values for each sets of scores in Z.
    each ROW of Z is a set of scores.
    """
    e_Z = np.exp(Z - np.max(Z, axis = 1, keepdims = True))
    A = e_Z / e_Z.sum(axis = 1, keepdims = True)
    return A

def one_hot(Y, num_classes=3):
    """
    Convert class labels to one-hot encoding.

    Parameters
    ----------
    Y : np.ndarray
        Vector of shape (N,), values in {0, 1, 2}
    num_classes : int
        Number of classes (default = 3)

    Returns
    -------
    np.ndarray
        One-hot encoded matrix of shape (N, 3)
    """
    Y = np.asarray(Y, dtype=int)
    return np.eye(num_classes)[Y]

def crossentropy_loss(Yhat, y):
    """
    Yhat: a numpy array of shape (Npoints, nClasses) -- predicted output
    y: a numpy array of shape (Npoints) -- ground truth.
    NOTE: We don't need to use the one-hot vector here since most of elements
    are zeros. When programming in numpy, in each row of Yhat, we need to access
    to the corresponding index only.
    """
    id0 = range(Yhat.shape[0])
    return -np.mean(np.log(Yhat[id0, y]))

def mlp_init(d0, d1, d2):
    """
    Initialize W1, b1, W2, b2
    d0: dimension of input data
    d1: number of hidden unit
    d2: number of output unit = number of classes
    """
    W1 = 0.01*np.random.randn(d0, d1)
    b1 = np.zeros(d1)
    W2 = 0.01*np.random.randn(d1, d2)
    b2 = np.zeros(d2)
    return (W1, b1, W2, b2)

def mlp_predict(X, W1, b1, W2, b2):
    """
    Suppose that the network has been trained, predict class of new points.
    X: data matrix, each ROW is one data point.
    W1, b1, W2, b2: learned weight matrices and biases
    """
    Z1 = X.dot(W1) + b1 # shape (N, d1)
    A1 = np.maximum(Z1, 0) # shape (N, d1)
    Z2 = A1.dot(W2) + b2 # shape (N, d2)
    return np.argmax(Z2, axis=1)






def my_neural_network(W15, b15, W25, b25,
                      nepoches = 2000,
                      batch_size = 3000,
                      lr=1):
    """
    X: N x d0
    Y: (N,) vector
    """

    N = X_train.shape[0]
    nbatches = int(np.ceil(float(N)/batch_size))

    # init
    W1 = W15.copy()
    b1 = b15.copy()
    W2 = W25.copy()
    b2 = b25.copy()

    accuracy_history = []
    loss_history = []
    for ep in range(nepoches):
        # mix_ids = np.random.permutation(N) # mix data
        mix_ids = np.arange(N)
        avarage_loss = 0
        for i in range(nbatches):
            batch_ids = mix_ids[batch_size*i:min(batch_size*(i+1), N)]
            X_batch, y_batch = X_train[batch_ids], y_train[batch_ids]

            Z1 = X_batch.dot(W1)+b1 # N x d1
            A1 = np.maximum(0, Z1) # N x d1
            Y_hat = softmax_stable(A1.dot(W2)+b2) # N x C

            avarage_loss += crossentropy_loss(Y_hat, y_batch)

            # backpropagation
            # E2 = (1/N)*(Y_hat-one_hot(y_batch, C)) # N x C
            id0 = range(Y_hat.shape[0])
            Y_hat[id0, y_batch] -=1
            E2 = Y_hat/batch_size # shape (N, d2)

            grad_W2 = A1.T.dot(E2)
            grad_b2 = E2.sum(axis=0)
            # E1 = (E2.dot(W2.T)) * ((Z1 > 0).astype(Z1.dtype)) # N x d1
            E1 = np.dot(E2, W2.T) # shape (N, d1)
            E1[Z1 <= 0] = 0 # gradient of ReLU, shape (N, d1)
            grad_W1 = X_batch.T.dot(E1)
            grad_b1 = E1.sum(axis=0)

            # update W and b
            W2 -= lr*grad_W2
            W1 -= lr*grad_W1
            b2 -= lr*grad_b2
            b1 -= lr*grad_b1
        y_pred = mlp_predict(X_test, W1, b1, W2, b2)
        accuracy_history.append(100*np.mean(y_pred == y_test))
        loss_history.append(avarage_loss/nbatches)
    return accuracy_history, loss_history


def neural_network_with_GDA(W15, b15, W25, b25,
                            nepoches = 2000,
                            batch_size = 3000,
                            lr=1,
                            my_sigma=0.1,
                            my_kappa=0.7):
    """
    X: N x d0
    Y: (N,) vector
    """
    my_lambda = lr

    N = X_train.shape[0]
    nbatches = int(np.ceil(float(N)/batch_size))

    # init
    W1 = W15.copy()
    b1 = b15.copy()
    W2 = W25.copy()
    b2 = b25.copy()

    accuracy_history = []
    loss_history = []
    for ep in range(nepoches):
        # mix_ids = np.random.permutation(N) # mix data
        mix_ids = np.arange(N)
        avarage_loss = 0
        for i in range(nbatches):
            batch_ids = mix_ids[batch_size*i:min(batch_size*(i+1), N)]
            X_batch, y_batch = X_train[batch_ids], y_train[batch_ids]

            Z1 = X_batch.dot(W1)+b1 # N x d1
            A1 = np.maximum(0, Z1) # N x d1
            Y_hat = softmax_stable(A1.dot(W2)+b2) # N x C

            current_loss = crossentropy_loss(Y_hat, y_batch)
            avarage_loss += current_loss

            # backpropagation
            id0 = range(Y_hat.shape[0])
            Y_hat[id0, y_batch] -=1
            E2 = Y_hat/batch_size # shape (N, d2)

            grad_W2 = A1.T.dot(E2)
            grad_b2 = E2.sum(axis=0)
            # E1 = (E2.dot(W2.T)) * ((Z1 > 0).astype(Z1.dtype)) # N x d1
            E1 = np.dot(E2, W2.T) # shape (N, d1)
            E1[Z1 <= 0] = 0 # gradient of ReLU, shape (N, d1)
            grad_W1 = X_batch.T.dot(E1)
            grad_b1 = E1.sum(axis=0)

            # update W and b
            W2 -= my_lambda*grad_W2
            W1 -= my_lambda*grad_W1
            b2 -= my_lambda*grad_b2
            b1 -= my_lambda*grad_b1

            Z1 = X_batch.dot(W1)+b1 # N x d1
            A1 = np.maximum(0, Z1) # N x d1
            Y_hat = softmax_stable(A1.dot(W2)+b2) # N x C

            sub = my_lambda*my_sigma*(np.sum(grad_W1**2)+np.sum(grad_W2**2)+np.sum(grad_b1**2)+np.sum(grad_b2**2))
            temp = current_loss - sub
            if ep==100 and i==20:
                print("epoch 200 and i = 20")
                print(my_lambda)
                print(crossentropy_loss(Y_hat, y_batch))
                print(current_loss)
                print(sub)
            if(crossentropy_loss(Y_hat, y_batch) > temp):
                my_lambda *= my_kappa

        y_pred = mlp_predict(X_test, W1, b1, W2, b2)
        accuracy_history.append(100*np.mean(y_pred == y_test))
        loss_history.append(avarage_loss/nbatches)
    return accuracy_history, loss_history




# suppose X, y are training input and output, respectively
NUM_OF_EPOCHES = 200
BATCH_SIZE = 50
TEST_SIZE = 500

SIGMA = 0.1 # sigma
KAPPA = 0.7
ETA = .5 # learning rate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
d0 = 2 # data dimension
d1 = h = 100 # number of hidden units
d2 = C = 3 # number of classes
(W1, b1, W2, b2) = mlp_init(d0, d1, d2)




accuracy_history, loss_history = my_neural_network(W1,
                                                   b1,
                                                   W2,
                                                   b2,
                                                   nepoches = NUM_OF_EPOCHES,
                                                   batch_size = BATCH_SIZE,
                                                   lr=ETA)

accuracy_history_gda = []
loss_history_gda = []
kappa_array = [0.7, 0.8, 0.9]

for i in range(3):
    temp1, temp2 = neural_network_with_GDA(W1,
                                                                    b1,
                                                                    W2,
                                                                    b2,
                                                                    nepoches = NUM_OF_EPOCHES,
                                                                    batch_size = BATCH_SIZE,
                                                                    lr=ETA,
                                                                    my_sigma=SIGMA,
                                                                    my_kappa=kappa_array[i])
    accuracy_history_gda.append(temp1)
    loss_history_gda.append(temp2)
    

with open("model_params_without_permutation.pkl", "wb") as f:
    pickle.dump((accuracy_history,
                 loss_history,
                 accuracy_history_gda,
                 loss_history_gda), f)


filename = 'neural_network_without_permutation.pdf'
with PdfPages(filename) as pdf:
    fig, ax = plt.subplots(1, 2, figsize=(8, 5), layout='constrained')
    ax[0].plot(np.arange(NUM_OF_EPOCHES), accuracy_history, label='GD')
    for i in range(3):
        ax[0].plot(np.arange(NUM_OF_EPOCHES), accuracy_history_gda[i], label='GDA_KAPPA='+str(kappa_array[i]))
    ax[0].set_xlabel('Number of iterations')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_title("Accuracy plot")
    ax[0].legend()

    ax[1].plot(np.arange(NUM_OF_EPOCHES), loss_history, label='GD')
    for i in range(3):
        ax[1].plot(np.arange(NUM_OF_EPOCHES), loss_history_gda[i], label='GDA_KAPPA='+str(kappa_array[i]))
    ax[1].set_xlabel('Number of iterations')
    ax[1].set_ylabel('Loss')
    ax[1].set_title("Loss plot")
    ax[1].legend()
    pdf.savefig(bbox_inches='tight')
    plt.show()