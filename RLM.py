import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as seabornInstance
from sklearn.model_selection import train_test_split  # , StratifiedKFold
from sklearn import linear_model
from sklearn import metrics
yc=0
data = pd.read_csv(r'yeni.csv')
Y = data['Y'].values
## Generate models 2^k-1
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
models =imprime(potencia(varind))
del models[0]
######################################################################

for i in range(0, len(models)):
    X = data[models[i]].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    regressor = linear_model.LinearRegression()
    regressor.fit(X_train, Y_train)
    y_pred = regressor.predict(X_test)
    df = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred})
    df1 = df.head(25)
    print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))
    print(regressor.intercept_)
    print(regressor.coef_)
    print(regressor.score(X_test, Y_test))
    df1.plot(kind='bar', figsize=(10, 8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()
    yp = []
    for z in range(0, len(Y)):
        for a in range(0, len(regressor.coef_)):
            # print (regressor.coef_[a],'*',data[models[i][a]].values[z])
            yc += regressor.coef_[a] * data[models[i][a]].values[z]
        yp = yp.append(regressor.intercept_ + yc)
    print(yp)