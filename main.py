import numpy as np
import numpy.linalg as la
import csv
import random


#recup la taille des données
#size = np.size(mat)[1]
#col1 = np.ones((size), dtype=int).reshape(size, 1)

#matriceFrut
#matFrut = [[64236.62],[4046]]


#x = np.arange(1, 301).reshape(300, 1)
#print("x")
#print(x)

#y = np.random.normal(x + 2, 50)

#col1 = np.ones((300), dtype=int).reshape(300, 1)
#print("les un")
#print(col1)

#x = np.concatenate((x, col1), axis=1)
#x = np.hstack((x, np.ones(x.shape)))
#print("concated")
#print(x)




def main():
    print("début du programme")

    # x = readCSV("frut_price_without_header.csv").astype(float)
    # print("le csv")
    # print(x)
    #
    # lenght = x.shape[0]
    # y = x[0:lenght, 0:1]
    # print("y")
    # print(y)
    #
    # x=np.delete(x,0,1)
    # x = np.hstack((x, np.ones(x.shape)))
    # print("x")
    # print(x)

    # x = csvToMatrixX()
    # y = csvToMatrixY()

    # matriceTrans = transverseMatrice(x)
    #print("transversed")
    #print(matriceTrans)
    # w = getW(x, matriceTrans, y)
    # print("w")
    #print(w)

    global X, Y, trans_X
    # csv_to_matrix("frut_price.csv")
    # print("y")
    # print(Y)

    X = np.arange(1, 301).reshape(300, 1)
    X = np.hstack((X, np.ones(X.shape)))

    y_c = np.random.normal(X + 2, 50)

    y_d = (y[i] <= 2 * x[i] + 3).astype(np.float32())

    trans_X = transpose_matrix(X)

    print("aprox")
    approximative_W = get_aproximative_w()
    print(approximative_W)

    print("exact")
    exact_W = get_exact_w(X, trans_X, Y)
    print(exact_W)
    #age = predictAge(w, x)
    #print (age)

def transpose_matrix(mat):
    trans_mat = np.transpose(mat)
    return trans_mat

def inverse_matrice(mat):
    inv_mat = la.inv(mat)
    return inv_mat

def transverseMatrice(mat):
    matTrans = np.transpose(mat)
    return matTrans

def inverseMatrice(mat):
    matInv = la.inv(mat)
    return matInv

def getW(x, xTrans, y):
    xTransXxInve = inverseMatrice(xTrans.dot(x))
    w = ((xTransXxInve.dot(xTrans)).dot(y))
    return w

def predictAge(w,x):
    return w.dot(x)

def readCSV(filename):
    reader = csv.reader(open(filename, "r"), delimiter=",")
    x = list(reader)
    result = np.array(x)
    return result

def csvToMatrix():
    my_data = np.loadtxt("frut_price.csv", delimiter=',', skiprows=1)
    Y = my_data[:,0]
    X = my_data[:,1:]
    col1 = np.full((len(Y), 1), 1)
    X = np.append(X, col1, axis=1)
    #print(Y)
    #print(X)

def csvToMatrixX():
    my_data = np.loadtxt("frut_price.csv", delimiter=',', skiprows=1)
    Y = my_data[:, 0]
    X = my_data[:,1:]
    col1 = np.full((len(Y), 1), 1)
    X = np.append(X, col1, axis=1)
    return X

def csvToMatrixY():
    my_data = np.loadtxt("frut_price.csv", delimiter=',', skiprows=1)
    Y = my_data[:,0]
    return Y

def get_aproximative_w():
    # Initialisation
    global X, Y
    W = np.matrix('0.1 0.2')
    alpha = 0.0000001
    # looping
    # print("shape x")
    # print(X.shape)
    # print("shape trans_X")
    # print(trans_X.shape)
    # print("shape W")
    # print(W.shape)
    # print("shape y")
    # print(Y.shape)
    for i in range(0, 2000):
        # picking random line in the matrix
        random_line = random.randrange(0, (len(Y) - 1))
        # print(X.shape)
        #W = W + (alpha * ((Y[random_line] - W.dot(X)) * X[random_line]))

        W = W - alpha * (((-2 / (i+1)) * trans_X * (Y - X.dot(W))))
    return W

def get_exact_w(X, trans_X, Y):
    trans_X_x_inv_X = inverse_matrice(trans_X.dot(X))
    W = ((trans_X_x_inv_X.dot(trans_X)).dot(Y))
    return W

def csv_to_matrix(filename):
    global X, Y
    csv_data = np.loadtxt(filename, delimiter=',', skiprows=1)
    Y = csv_data[:,0]
    X = csv_data[:,1:]
    col1 = np.full((len(Y), 1), 1)
    X = np.append(X, col1, axis=1)

##FONCTION PROF ##
def gradient_step(X, y, W):
    n = X.shape[0]
    W = W - alpha * (-2/n) * np.dot(X.t, y - g(X))
    return W

def gradient_step(X, y, W):
    n = X.shape[0]
    W = W - alpha * (-2/n) * np.dot(X, y - g(X))
    return W

def g(X, W):
    return np.dot(X, W)

def g(X):
    return np.dot(X, W_iter)

def gradient_descent(X, Y, alpha = 0.01, nb_iter = 1500):
    W_iter = np.random.uniform(-1, 1, (X.shape[1], 1))
    for i in range(nb_iter):
        print(W_iter)
        W_iter = gradient_step(W_iter)

main()