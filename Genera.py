## Generate models 2^k-1
varind = ['X1', 'X2']
model=[]
def potencia(c):
    if len(c) == 0:
        return[[]]
    a = potencia(c[:-1])
    return a+[s+[c[-1]] for s in a]
def imprime(c):
    for e in sorted(c, key=lambda s: (len(s), s)):
        model.append("','".join(e))
    return model
models =imprime( potencia(varind))
del models[0]

print (models)