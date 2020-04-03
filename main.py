import numpy as np
import numpy.linalg as la
import random
from math import exp
import math

X = None
Y = None

def initMatrix():
    global X, Y
    X = np.arange(1, 301).reshape(300, 1)
    #Y = np.random.normal(X + 2, 50)
    d = (-2*X + 600)
    Y = (X <= d).astype(np.float32())
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

def sigmoid(arrayOfN):
    for n in arrayOfN:
        arrayOfN[0] = 1 / (1 + math.exp(-n))
    return arrayOfN

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
        # if i % 10 == 0:
        #     print(random_line)
        #     print("W" + str(i) + ": " + str(W))
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
    print(X.shape)
    trans_X = transpose_matrix(X)
    W = np.matrix('0.25676164; 0.83605127')
    alpha = 0.00001
    # looping
    for i in range(0, 1001):
        W = W - alpha * (((1 / len(Y)) * trans_X.dot(sigmoid(g(X, W)) - Y)))
        print("aaa")
        print(sum(abs(np.round(sigmoid(g(X,W)))-Y)))
        # displaying every 10 iterations
        #if i % 100 == 0:
            #print("*****sigmoid = " + str(sigmoid(g(X, W))))
            #print("W" + str(i) + ": " + str(W))
    print("ICI !!!")
    print(sigmoid(g(X[2], W)))
    print(Y[2])
    return W


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
    #print(X)
    #print(Y)
    # Transposing X
    trans_X = transpose_matrix(X)
    # Getting app W

    approximative_W_sigmoid = get_aproximative_w_sigmoid()
    print("Approximative W (Sigmoid) = " + str(approximative_W_sigmoid))
    # approximative_W_rosenblatt = get_aproximative_w_rosenblatt()
    # print("Approximative W (Rosenblatt) = " + str(approximative_W_rosenblatt))
    # Getting app W
    # approximative_W = get_aproximative_w()
    # print("Approximative W = " + str(approximative_W))
    # Getting exact W
    #exact_W = get_exact_w(X, trans_X, Y)
    #print("Exact W = " + str(exact_W))
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