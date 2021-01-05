import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as seabornInstance
from sklearn.model_selection import train_test_split  # , StratifiedKFold
from sklearn import linear_model
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
yc = 0
r2 = 0
data = pd.read_csv(r'yeni.csv')
Y = data['Y'].values
X = data[['X1', 'X2', 'X4', 'X5']].values
######################################################################
regressor = MLPRegressor(hidden_layer_sizes=(10,5,), activation='relu',
                                          solver='adam', alpha=1e-7, batch_size='auto', learning_rate='adaptive',
                                          learning_rate_init=0.001, power_t=0.9, max_iter=10000000, shuffle=True, random_state=None,
                                          tol=1e-4, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                                          early_stopping=False, validation_fraction=0.1, epsilon=1e-08,
                                          n_iter_no_change=10, max_fun=15000)
######################################################################


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
regressor = regressor.fit(X_train, Y_train)
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred})
df1 = df.head(25)
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))
print(regressor.score(X_test, Y_test))
df1.plot(kind='bar', figsize=(10, 8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
allpred = regressor.predict(X)

