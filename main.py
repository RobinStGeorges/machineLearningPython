import numpy as np
import numpy.linalg as la
import csv


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

    x = csvToMatrixX()
    y = csvToMatrixY()

    print(y)

    # matriceTrans = transverseMatrice(x)
    #print("transversed")
    #print(matriceTrans)
    # w = getW(x, matriceTrans)
    #print("w")
    #print(w)
    #age = predictAge(w, x)
    #print (age)


def transverseMatrice(mat):
    matTrans = np.transpose(mat)
    return matTrans

def inverseMatrice(mat):
    matInv = la.inv(mat)
    return matInv

def getW(x, xTrans):
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
    print(Y)
    print(X)

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



main()