import numpy as np
import numpy.linalg as la

x = np.arange(1, 301).reshape(300, 1)
print(x)
col1 = np.ones((300), dtype=int).reshape(300, 1)
print(col1)
x = np.concatenate((x, col1), axis=0)

print(x)


y = np.random.normal(x + 2, 50)

def main():
    print("début du programme")
    matriceTrans = transverseMatrice(x)
    w = getW(x, matriceTrans)


def transverseMatrice(mat):
    print("avant transverse")
    print(mat)
    print("après transverse")
    matTrans = np.transpose(mat)
    print (matTrans)
    return matTrans

def inverseMatrice(mat):
    print("avant inverse")
    print(mat)
    print("après inverse")
    matInv = la.inv(mat)
    print(matInv)
    return matInv

def getW(x, xTrans):
    print("getW")
    xTransXxInve = inverseMatrice(xTrans.dot(x))
    w = ((xTransXxInve.dot(xTrans)).dot(y))
    print("et w :")
    print(w)
    return w



main()