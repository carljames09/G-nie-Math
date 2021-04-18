# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 08:40:14 2021

@author: mdr13
"""

import numpy as np
import time as t
import matplotlib.pyplot as plt

#----------------------------------Cholesky------------------------------------
def Cholesky(A):
    n,m = np.shape(A)
    L = np.zeros((n,m))
    L[0,0] = (A[0][0])**0.5
    for i in range(n):
        for j in range(i+1):
            S1 = 0
            S2 = 0
            L[i][0] = A[i][0] / L[0][0]
            for k in range(j):
                S1 = S1 + L[j][k]**2
                S2 = S2 + L[i][k]*L[j][k]
                if S1 == 0:
                    return "Matrice A non positive"
                else:
                    L[j][j] = (A[j][j] - S1)**0.5
                    L[i][j] = (A[i][j] - S2) / L[j][j]
    return L

def ResolutionSysTriSup(Taug):  
    n,m = np.shape(Taug)
    x = np.zeros(n)
    for i in range (n-1, -1, -1):
        somme = 0
        for k in range (i+1, n):
            somme = somme + x[k] * Taug[i,k]
        x[i] = (Taug[i,n] - somme) / Taug[i,i]  
    return x

def ResolutionSysTriInf(A):
    n,m = np.shape(A)
    x = np.zeros(n)
    for i in range(0,n):
        somme = 0
        for k in range(0,i):
            somme = somme + A[i,k] * x[k]
        x[i] = (A[i,n] - somme) / A[i,i]
    return x

def ResolCholesky (A, B):
    t1 = t.time()
    L = Cholesky (A)
    print(L)
    Taug = np.c_[L, B]
    Y = ResolutionSysTriInf (Taug)
    Y = Y[:, np.newaxis]
    LT = np.transpose(L)
    Baug = np.c_[LT, Y]
    X = ResolutionSysTriSup (Baug)
    t2 = t.time()
    t_final = t2 - t1
    return X, t_final
#----------------------------Linalg Cholesky-----------------------------------
def SolveurLinalgChol(A, B):
    t1 = t.time()
    L = np.linalg.cholesky(A)
    Taug = np.c_[L, B]
    Y = ResolutionSysTriInf (Taug)
    Y = Y[:, np.newaxis]
    LT = np.transpose(L)
    Baug = np.c_[LT, Y]
    X = ResolutionSysTriSup (Baug)
    t2 = t.time()
    t_final = t2 - t1
    return X, t_final
#----------------------------------Gauss---------------------------------------
def ReductionGauss(Aaug):
    tup = Aaug.shape
    n = tup[0]
    for i in range(0, n-1):
        for k in range(i+1, n):
            g = Aaug[k, i] / Aaug[i, i]
            for j in range(i, n+1):
                Aaug[k, j] = Aaug[k, j] - g * Aaug[i, j]
    return Aaug

def Gauss(A, B):
    Aaug = np.c_[A, B]
    t1 = t.time()
    Taug = ReductionGauss(Aaug)
    #print(Taug)
    x = ResolutionSysTriSup(Taug)
    t2 = t.time()
    t_final = t2 - t1
    return x, t_final

def Erreur(A, B, X):
    vec = np.dot(A, X) - B
    return np.linalg.norm(vec)
"""
#-------------------------------------Test-------------------------------------
M = np.random.randint(low=1, high=10, size=(53,53))
MT = np.transpose(M)
A = np.dot(M,MT)
B = np.random.randint(low=1, high=10, size=53)
print("A=",A)
print("B=",B)
sol,temps = ResolCholesky(A, B)
erreur = Erreur(A, B, sol)

print("X=",sol)
print(temps, erreur)
print("#################################################")
x,t1 = SolveurLinalgChol(A, B)
err = Erreur(A, B, x)
print("X linalg=",x)
print(t1,err)
"""
#--------------------------------Graphes---------------------------------------
TChol = list ()
EChol = list ()
NChol = list ()

TGauss = list ()
EGauss = list ()
NGauss = list ()

TLinalg = list ()
ELinalg = list ()
NLinalg = list ()

 
for n in range(10, 300, 10):
    A = np.random.rand(n,n)
    B = np.random.rand(n)
    sol, temps = Gauss(A, B)
    erreur = Erreur(A, B, sol)
    NGauss.append(n)
    TGauss.append(temps)
    EGauss.append(erreur)
    
for n in range (10, 300, 10):
    M = np.random.randint(low=1, high=10, size=(n,n))
    MT = np.transpose(M)
    A = np.dot(M,MT)
    B = np.random.randint(low=1, high=10, size=n)
    sol, temps = ResolCholesky(A, B)
    erreur = Erreur (A, B, sol)
    NChol.append(n)
    TChol.append(temps)
    EChol.append(erreur)
    
for n in range (10, 300, 10):
    M = np.random.randint(low=1, high=10, size=(n,n))
    MT = np.transpose(M)
    A = np.dot(M,MT)
    B = np.random.randint(low=1, high=10, size=n)
    sol, temps = SolveurLinalgChol (A, B)
    erreur = Erreur (A, B, sol)
    NLinalg.append(n)
    TLinalg.append(temps)
    ELinalg.append(erreur)
    
def GraphTemps():
    plt.figure ()
    plt.subplot()
    plt.plot (NChol, TChol, color ='green', label = 'ResolCholesky')
    plt.plot (NGauss, TGauss, color = 'blue', label = 'Gauss')
    plt.plot (NLinalg, TLinalg, color = 'red', label = 'LinalgCho')
    plt.legend()
    plt.xlabel('Dimension')
    plt.ylabel('Temps (en s)')
    plt.title('Temps de calcul en fonction de la dimension')    
    plt.show()
    
def GraphErreur():
    plt.figure ()
    plt.subplot ()
    plt.plot (NChol, EChol, color = 'green', label = 'ResolChelesky')
    plt.plot (NGauss, EGauss, color = 'blue', label='Gauss')
    plt.plot (NLinalg, ELinalg, color = 'red', label = 'Linalg')
    plt.legend ()
    plt.xlabel ('Dimension')
    plt.ylabel ('Erreur (norme)')
    plt.title ('Erreur de calcul en fonction de la dimension')
    plt.show()

    
GraphTemps()
GraphErreur() 