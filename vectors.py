import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from sympy import symbols, Eq, solve
import matplotlib.pyplot as plt

P, Q , R = symbols('P, Q, R')
X = np.array([P,Q,R])
print(X)

r1 = np.array([1,1,0])
r2 = np.array([0,1,1])
r3 = np.array([1,0,1])
I = np.array([r1,r2,r3])
print(I)
ivr =np.linalg.inv(I)
print(ivr )

T = np.matmul(ivr,X)
print(T)

P = np.array([3,4])
Q = np.array([4,6])
R = np.array([5,7])

A =2*(0.5*P - 0.5*Q + 0.5*R)
print(A)
A =2*(0.5*P + 0.5*Q - 0.5*R)
print(A)
A =2*(-0.5*P + 0.5*Q + 0.5*R)
print(A)

A1 = np.array([4,5])
B1 = np.array([2,3])
C1 = np.array([6,9])

def dir_vec(A,B):
  return B-A

def norm_vec(A,B):
  return omat*dir_vec(A,B)


def line_gen(A,B):
  len =10
  x_AB = np.zeros((2,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB

A = np.array([-2,-2]) 
B = np.array([1,3]) 
dvec = np.array([-1,1]) 
omat = np.array([[0,1],[-1,0]])

x_AB = line_gen(A1,B1)
x_BC = line_gen(B1,C1)
x_AC = line_gen(A1,C1)


#Triangle
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_AC[0,:],x_AC[1,:],label='$AC$')
plt.plot(A1[0], A1[1], 'o')
plt.text(A1[0] * (1 + 0.1), A1[1] * (1 - 0.1) , 'A')
plt.plot(B1[0], B1[1], 'o')
plt.text(B1[0] * (1 - 0.2), B1[1] * (1) , 'B')
plt.plot(C1[0], C1[1], 'o')
plt.text(C1[0] * (1 - 0.2), C1[1] * (1) , 'C')


#Orthocenters
plt.plot(P[0], P[1], 'o')
plt.text(P[0] * (1 + 0.1), P[1] * (1 - 0.1) , 'P')
plt.plot(R[0], R[1], 'o')
plt.text(R[0] * (1 - 0.2), R[1] * (1) , 'R')
plt.plot(Q[0], Q[1], 'o')
plt.text(Q[0] * (1 - 0.2), Q[1] * (1) , 'Q')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.axis('equal')
plt.grid()
plt.show()
