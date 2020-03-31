import csv
import numpy as np
import numpy.linalg as la


# recup la taille des données
#size = np.size(mat)[1]
#col1 = np.ones((size), dtype=int).reshape(size, 1)

# matriceFrut
matFrut = [[64236.62], [4046]]


x = np.arange(1, 301).reshape(300, 1)
print("x")
# print(x)

col1 = np.ones((300), dtype=int).reshape(300, 1)
print("les un")
# print(col1)

x = np.concatenate((x, col1), axis=1)
print("concated")
print(x)


y = np.random.normal(x + 2, 50)


def main():
    print("début du programme")
    matriceTrans = transverseMatrice(x)
    print("transversed")
    print(matriceTrans)
    w = getW(x, matriceTrans)
    print("w")
    print(w)
    #age = getAge(w, x)
    #print (age)
    readCSV("frut_price.csv")


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


def getAge(w, x):
    return w.dot(x)


def readCSV(filename):
    reader = csv.reader(open(filename, "rb"), delimiter=",")
    x = list(reader)
    result = np.array(x).astype("string")
    return result


main()
