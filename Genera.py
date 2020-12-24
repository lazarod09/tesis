## Generate models 2^k-1
import itertools

varind = ['X1', 'X2', 'X3']
model=[]
def potencia(c):
    if len(c) == 0:
        return[[]]
    a = potencia(c[:-1])
    return a+[s+[c[-1]] for s in a]
def imprime(c):
    for e in sorted(c, key=lambda s: (len(s), s)):
        model.append(e)
    return model
X =imprime( potencia(varind))
del X[0]
print ( X)