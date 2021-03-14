# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 09:51:54 2021

@author: mdr13
"""


import numpy as np
import time as t
import matplotlib.pyplot as plt

#--------------------Gauss sans changement de pivot----------------------------
def ReductionGauss(Aaug):
    tup = Aaug.shape
    n = tup[0]
    for i in range(0, n-1):
        for k in range(i+1, n):
            g = Aaug[k, i] / Aaug[i, i]
            for j in range(i, n+1):
                Aaug[k, j] = Aaug[k, j] - g * Aaug[i, j]
    return Aaug

def ResolutionSystTriSup(Taug):  
    n,m = np.shape(Taug)
    x = np.zeros(n)
    for i in range (n-1, -1, -1):
        somme = 0
        for k in range (i, n):
            somme = somme + x[k] * Taug[i,k]
            x[i] = (Taug[i,n] - somme) / Taug[i,i]  
    return x

def Gauss(A, B):
    Aaug = np.c_[A, B]
    t1 = t.time()
    Taug = ReductionGauss(Aaug)
    #print(Taug)
    x = ResolutionSystTriSup(Taug)
    t2 = t.time()
    t_final = t2 - t1
    return x, t_final

#-----------------------------Décomposition LU---------------------------------
def DecompositionLU(A):
    n,m = np.shape(A)
    U = np.zeros((n,m))
    L = np.eye(n)
    for i in range(0,n):
        for k in range(i+1,n):
            g = A[i][i]
            g = A[k][i]/g
            L[k][i] = g
            for j in range(i,n):
                A[k][j] = A[k][j] - g * A[i][j]
    U = A
    return L, U

def ResolutionSysTriInf(A):
    n,m = np.shape(A)
    x = np.zeros(n)
    for i in range(0,n):
        somme = 0
        for k in range(0,i):
            somme = somme + A[i,k] * x[k]
        x[i] = (A[i,n] - somme) / A[i,i]
    return x

def ResolutionLU(A,B):
    n,m = np.shape(A)
    x = np.zeros(n)
    t1 = t.time()
    L,U = DecompositionLU(A)
    Y = ResolutionSysTriInf(np.c_[L,B])
    Y1 = np.asarray(Y).reshape(n,1)
    x = ResolutionSystTriSup(np.c_[U,Y1])
    t2 = t.time()
    t_final = t2 - t1
    return x, t_final

#---------------------------Gauss avec pivot partiel---------------------------
def ReductionGaussPartiel(Aaug) :
    n,m = Aaug.shape
    #print(Aaug)
    for k in range(n-1):
        pivot_max = Aaug[k,k]
        indice_max = k
        for i in range(k+1,n): #balayage ligne
            if abs(Aaug[k,k]) > abs(pivot_max):
                pivot_max = Aaug[i,i]
                indice_max = i
        K= np.copy(Aaug[k,:])
        Aaug[k,:] = Aaug[indice_max,:]
        Aaug[indice_max,:] = K
        #print(Aaug)
        for i in range (k+1,n):
            g = Aaug[i,k] / Aaug[k,k]
            for j in range(k, n+1):
                Aaug[i,j] = Aaug[i,j] - g * Aaug[k,j]
    return Aaug

def GaussChoixPivotPartiel(A,B):
    Aaug = np.c_[A, B]
    t1 = t.time()
    Taug = ReductionGaussPartiel(Aaug)       
    x = ResolutionSystTriSup(Taug)
    t2 = t.time()
    t_final = t2 - t1 
    return x, t_final

#---------------------------Gauss avec pivot total-----------------------------
def ReductionGaussTotal(Aaug) :
    n,m = Aaug.shape
    for k in range(n):
        #print(Aaug)
        pivot_max = Aaug[k,k]
        indice_max = k
        for i in range(k+1,n): #balayage ligne
            if abs(Aaug[i,i]) > abs(pivot_max):
                pivot_max = Aaug[i,i]
                indice_max = i
        K= np.copy(Aaug[k,:])
        Aaug[k,:] = Aaug[indice_max,:]
        Aaug[indice_max,:] = K
        #print(Aaug)
        for i in range(k+1,m):
            if abs(Aaug[k,k]) <= abs(Aaug[k,i]):
                K = np.copy(Aaug[:,k])
                Aaug[:,k] = Aaug[:,i]
                Aaug[:,i] = K
        #print('######')
        #print(Aaug)
        #print('######')
        for i in range (k+1,n):
           g = Aaug[i,k]/Aaug[k,k]
           #print(g)
           for j in range(k, n+1):
               Aaug[i,j] = Aaug[i,j] - g * Aaug[k,j]
              #print(Aaug)
    return Aaug

def GaussChoixPivotTotal(A, B):
    Aaug = np.c_[A, B]
    t1 = t.time()
    Taug = ReductionGaussTotal(Aaug)
    x = ResolutionSystTriSup(Taug)
    t2 = t.time()
    t_final = t2 - t1
    return x, t_final
#--------------------Comparaison avec linalg.solve-----------------------------
def SolveurLinalg(A, B):
    t1 = t.time()
    x = np.linalg.solve(A, B)
    t2 = t.time()
    t_final = t2 - t1
    return x, t_final

def Erreur(A, B, X):
    vec = np.dot(A, X) - B
    return np.linalg.norm(vec)

def Mat_Random(n):
    A = np.random.rand(n, n)
    B = np.random.rand(n)
    return A, B

def Mat_RandomInt(n):
    A = np.random.randint(low=1, high=100, size=(n, n))
    B = np.random.randint(low=1, high=100, size=n)
    return A, B
"""
#-----------------------------------Test---------------------------------------
A = np.array([[3,5,-1,4], [-3,-6,4,-2], [6,2,2,7], [9,4,2,18]])
B = np.array([4,-5,-2,13])
x1,t1 = GaussChoixPivotPartiel(A, B)
x2,t2 = Gauss(A, B)
x3,t3 = GaussChoixPivotTotal(A, B)
x4,t4 = ResolutionLU(A, B)
print(x1)
print(x2)
print(x3)
print(x4)

"""

NGauss = list()
TGauss = list()
EGauss = list()

NLU = list()
TLU = list()
ELU = list()

NPartiel = list()
TPartiel = list()
EPartiel = list()

NTotal = list()
TTotal = list()
ETotal = list()

NLinal = list()
TLinal = list()
ELinal = list()

for n in range(10, 200, 20):
    A, B = Mat_Random(n)
    #print(A)
    #print("------------------------------------")
    #print(B)
    #print("------------------------------------")
    sol, temps = Gauss(A, B)
    erreur = Erreur(A, B, sol)
    #print(sol)
    #print(temps)
    #print(Erreur(A, B, sol))
    NGauss.append(n)
    TGauss.append(temps)
    EGauss.append(erreur)

for n in range(10, 200, 20):
    A, B = Mat_Random(n)
    sol, temps = ResolutionLU(A, B)
    erreur = Erreur(A, B, sol)
    NLU.append(n)
    TLU.append(temps)
    ELU.append(erreur)
    
for n in range(10, 200, 20):
    A, B = Mat_Random(n)
    sol, temps = GaussChoixPivotPartiel(A, B)
    erreur = Erreur(A, B, sol)
    NPartiel.append(n)
    TPartiel.append(temps)
    EPartiel.append(erreur)

for n in range(10, 200, 20):
    A, B = Mat_Random(n)
    sol, temps = GaussChoixPivotTotal(A, B)
    erreur = Erreur(A, B, sol)
    NTotal.append(n)
    TTotal.append(temps)
    ETotal.append(erreur)
    
for n in range(10, 200, 20):
    A, B = Mat_Random(n)
    sol, temps = SolveurLinalg(A, B)
    erreur = Erreur(A, B, sol)
    NLinal.append(n)
    TLinal.append(temps)
    ELinal.append(erreur)
    
def GraphTemps():    
    plt.figure()
    plt.subplot()
    plt.plot(NGauss, TGauss, color = 'blue', label='Gauss')
    plt.plot(NLU, TLU, color='black', label='LU')
    plt.plot(NPartiel, TPartiel, color = 'green', label='Partiel')
    plt.plot(NTotal, TTotal, color = 'red', label='Total')
    plt.plot(NLinal, TLinal, color = 'orange', label='Linalg.solve')
    plt.legend()
    plt.xlabel('Dimension')
    plt.ylabel('Temps (en s)')
    plt.title('Temps de calcul en fonction de la dimension')    
    plt.show()
    
def GraphTempslog():
    plt.figure()
    plt.plot(NGauss, TGauss, color = 'blue', label='Gauss')
    plt.plot(NLU, TLU, color='black', label='LU')
    plt.plot(NPartiel, TPartiel, color = 'green', label='Partiel')
    plt.plot(NTotal, TTotal, color = 'red', label='Total')
    plt.plot(NLinal, TLinal, color = 'orange', label='Linalg.solve')
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Dimension')
    plt.ylabel('Temps (en s)')
    plt.title('Temps de calcul en fonction de la dimension')
    plt.show()
    
def GraphErreur():
    plt.figure()
    plt.subplot()
    plt.plot(NGauss, EGauss, color = 'blue', label='Gauss')
    plt.plot(NLU, ELU, color='black', label='LU')
    plt.plot(NPartiel, EPartiel, color = 'green', label='Partiel')
    plt.plot(NTotal, ETotal, color = 'red', label='Total')
    plt.plot(NLinal, ELinal, color = 'orange', label='Linalg.solve')
    plt.legend()
    plt.xlabel('Dimension')
    plt.ylabel('Erreur (norme)')
    plt.title('Erreur de calcul en fonction de la dimension')
    plt.show()
    
GraphTemps()
GraphTempslog()
GraphErreur()
