import numpy as np
import numpy.linalg as la
import random
from math import exp

X = None
Y = None

def initMatrix():
    global X, Y
    X = np.arange(1, 301).reshape(300, 1)
    Y = np.random.normal(X + 2, 50)
    col1 = np.full((len(Y), 1), 1)
    X = np.append(X, col1, axis=1)

def csv_to_matrix(filename):
    global X, Y
    csv_data = np.loadtxt(filename, delimiter=',', skiprows=1)
    Y = csv_data[:,0]
    X = csv_data[:,1:]
    col1 = np.full((len(Y), 1), 1)
    X = np.append(X, col1, axis=1)

def transpose_matrix(mat):
    trans_mat = np.transpose(mat)
    return trans_mat

def inverse_matrice(mat):
    inv_mat = la.inv(mat)
    return inv_mat

def g(X, W):
    return np.dot(X, W)

def sigmoid(n):
    return (1 / (1 + exp(-n)))

def get_exact_w(X, trans_X, Y):
    trans_X_x_inv_X = inverse_matrice(trans_X.dot(X))
    W = ((trans_X_x_inv_X.dot(trans_X)).dot(Y))
    return W

def get_aproximative_w():
    # Initialization
    global X, Y
    W = np.matrix('0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9')
    alpha = 0.0000000000000001
    # looping
    for i in range(0, 1001):
        # picking random line in the matrix
        random_line = random.randrange(0, (len(Y) - 1))
        W = W + (alpha * (Y[random_line] - W.dot(X[random_line])) * X[random_line])
        # displaying every 10 iterations
        if i % 10 == 0:
            print(random_line)
            print("W" + str(i) + ": " + str(W))
    return W

def get_aproximative_w_rosenblatt():
    # Initialization
    global X, Y
    trans_X = transpose_matrix(X)
    W = np.matrix('0.25676164; 0.83605127')
    alpha = 0.0000001
    # looping
    for i in range(0, 1001):
        W = W - alpha * (((-2 / len(Y)) * trans_X.dot(Y - g(X, W))))
        # displaying every 10 iterations
        if i % 100 == 0:
            print("W" + str(i) + ": " + str(W))
    return W

def get_aproximative_w_sigmoid():
    # Initialization
    global X, Y
    trans_X = transpose_matrix(X)
    W = np.matrix('0.25676164 0.83605127')
    trans_W = transpose_matrix(W)
    alpha = 0.00000001
    # looping
    for i in range(0, 2001):
        W = W - alpha * (((-2 / (i+1)) * trans_X * (Y - sigmoid(X.dot(trans_W)))))
        # displaying every 10 iterations
        if i % 100 == 0:
            print("W" + str(i) + ": " + str(W))
    return W

# def gradient_step(X, y, W):
#     n = X.shape[0]
#     W = W - alpha * (-2/n) * np.dot(X.t, y - g(X))
#     return W

# def gradient_descent(X, Y, alpha = 0.01, nb_iter = 1500):
#     W_iter = np.random.uniform(-1, 1, (X.shape[1], 1))
# 
#     for i in range(nb_iter):
#         print(W_iter)
#         W_iter = gradient_step(W_iter)


def predictAveragePrice(W, X):
    return W.dot(X)

def classification():
    global X, Y
    X = np.arange(1, 301).reshape(300, 1)
    YC = np.random.normal(X + 2, 50)
    YD = (YC[i]<= -0.5*X[i] + 300).astype(np.float32())
    mu = np.mean(X, 0)
    sigma = np.std(X, 0)
    X = (X-mu) / sigma

def main():
    print("début du programme")
    # Fetching mat X and Y
    global X, Y
    # csv_to_matrix("frut_price.csv")
    initMatrix()
    print(X)
    print(Y)
    # Transposing X
    trans_X = transpose_matrix(X)
    # Getting app W
    # approximative_W_sigmoid = get_aproximative_w_sigmoid()
    # print("Approximative W (Sigmoid) = " + str(approximative_W_sigmoid))
    approximative_W_rosenblatt = get_aproximative_w_rosenblatt()
    print("Approximative W (Rosenblatt) = " + str(approximative_W_rosenblatt))
    # Getting app W
    # approximative_W = get_aproximative_w()
    # print("Approximative W = " + str(approximative_W))
    # Getting exact W
    exact_W = get_exact_w(X, trans_X, Y)
    print("Exact W = " + str(exact_W))
    # Applying fouded Ws to a line in order to predict age
    # line = 5
    # print("Price in line " + str(line) + " = " + str(Y[line]))
    # predictedPriceWithApproximativeW = predictAveragePrice(approximative_W, X[line])
    # print("Predicted price with approximative W = " + str(predictedPriceWithApproximativeW))
    # predictedPriceWithExactW = predictAveragePrice(exact_W, X[line])
    # print("Predicted price with exact W = " + str(predictedPriceWithExactW))
    # print("fin du programme")

# Lunching the program
main()