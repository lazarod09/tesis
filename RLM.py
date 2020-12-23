import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as seabornInstance
from sklearn.model_selection import train_test_split  # , StratifiedKFold
from sklearn import linear_model
from sklearn import metrics

data = pd.read_csv(r'yeni.csv')

## Generate models 2^k-1
vardep='x1~'
varind = ['t2','t3']
model=[]
def potencia(c):
    if len(c) == 0:
        return[[]]
    a = potencia(c[:-1])
    return a+[s+[c[-1]] for s in a]
def imprime(c):
    for e in sorted(c, key=lambda s: (len(s), s)):
        model.append(prefix +" +".join(e))
    return model
models =imprime( potencia(varind))
del models[0]
######################################################################


X = data[['X1', 'X2', 'X3', 'X4', 'X5']].values
Y = data['Y'].values

# skf = StratifiedKFold(n_splits=2)
# skf.get_n_splits(X, Y)
# for train_index, test_index in skf.split(X, Y):
#    print("TRAIN:", train_index, "TEST:", test_index)
#    X_train, X_test = X[train_index], X[test_index]
#    Y_train, Y_test = Y[train_index], Y[test_index]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
regressor = linear_model.LinearRegression()
regressor.fit(X_train, Y_train)

y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred})
df1 = df.head(25)

print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))

df1.plot(kind='bar', figsize=(10, 8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
print(regressor.intercept_)
print(regressor.coef_)
print(regressor.score(X_test, Y_test))