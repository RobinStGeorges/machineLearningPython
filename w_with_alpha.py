import numpy as np
import numpy.linalg as la
import random

X = None
Y = None

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

def get_exact_w(X, trans_X, Y):
    trans_X_x_inv_X = inverse_matrice(trans_X.dot(X))
    W = ((trans_X_x_inv_X.dot(trans_X)).dot(Y))
    return W

def get_aproximative_w():
    # Initialisation
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

def predictAveragePrice(W, X):
    return W.dot(X)

def main():
    print("début du programme")
    # Fetching mat X and Y
    global X, Y
    csv_to_matrix("frut_price.csv")
    print(X)
    print(Y)
    # Transposing X
    trans_X = transpose_matrix(X)
    # Getting app W
    approximative_W = get_aproximative_w()
    print("Approximative W = " + str(approximative_W))
    # Getting exact W
    exact_W = get_exact_w(X, trans_X, Y)
    print("Exact W = " + str(exact_W))
    # Applying fouded Ws to a line in order to predict age
    line = 5
    print("Price in line " + str(line) + " = " + str(Y[line]))
    predictedPriceWithApproximativeW = predictAveragePrice(approximative_W, X[line])
    print("Predicted price with approximative W = " + str(predictedPriceWithApproximativeW))
    predictedPriceWithExactW = predictAveragePrice(exact_W, X[line])
    print("Predicted price with exact W = " + str(predictedPriceWithExactW))
    print("fin du programme")

# Lunching the program
main()