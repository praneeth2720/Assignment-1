import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from sympy import symbols, Eq, solve
import matplotlib.pyplot as plt

X1 = np.array([3,4])
X2 = np.array([4,6])
X3 = np.array([5,7])

x1, x2 , x3 = symbols('x1, x2, x3')
# defining equations
eq1 = Eq((x1+x2), 6)
print("Equation 1:")
print(eq1)
eq2 = Eq((x1+x3), 10)
print("Equation 2")
print(eq2)
eq3 = Eq((x2+x3), 8)
print("Equation 3")

y1, y2 , y3 = symbols('y1, y2, y3')
# defining equations
eq4 = Eq((y1+y2), 8)
print("Equation 4:")
print(eq1)
eq5 = Eq((y1+y3), 14)
print("Equation 5")
print(eq2)
eq6 = Eq((y2+y3), 12)
print("Equation 6")

print("Values of 3 unknown variable are as follows:")
print(solve((eq1, eq2, eq3, eq4, eq5, eq6), (x1, y1, x2, y2, x3, y3)))

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
plt.text(A1[0] * (1 + 0.1), A1[1] * (1 - 0.1) , 'A1')
plt.plot(B1[0], B1[1], 'o')
plt.text(B1[0] * (1 - 0.2), B1[1] * (1) , 'B1')
plt.plot(C1[0], C1[1], 'o')
plt.text(C1[0] * (1 - 0.2), C1[1] * (1) , 'C1')

#Orthocenters
plt.plot(X1[0], X1[1], 'o')
plt.text(X1[0] * (1 + 0.1), X1[1] * (1 - 0.1) , 'X1')
plt.plot(X2[0], X2[1], 'o')
plt.text(X2[0] * (1 - 0.2), X2[1] * (1) , 'X2')
plt.plot(X3[0], X3[1], 'o')
plt.text(X3[0] * (1 - 0.2), X3[1] * (1) , 'X3')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.axis('equal')
plt.grid()
plt.show()
