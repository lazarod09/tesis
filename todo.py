# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 16:00:44 2021

@author: Miguel
"""
#####################################################
#                  OLS
#####################################################
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats import outliers_influence as oi
import statsmodels.tools as st
import pandas as pd  # Import XLS File
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
from sklearn import metrics as skm
#####################################################
#               SKLEARN
#####################################################
from sklearn.model_selection import train_test_split  # , StratifiedKFold
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import statistics as stat
 
#####################################################
#               MARS
#####################################################
from pyearth import Earth
#####################################################
# Load Data
data = pd.read_excel(r'BD.xls', sheet_name='RLuna')
X = data[['X1','X2','X4','X5']].values
y = data['Y'].values
#################################
#####################################################
sign = 0.05
#####################################################
#                  OLS
#####################################################
XX = sm.add_constant(X)
met1= sm.OLS(y, XX).fit()
# Seria Mejor llamar al que dice OLS-Modif
#####################################################
print(met1.summary())
ols_y= met1.predict(XX)
ols_resid = y - ols_y
#####################################################
#               SKLEARN
#####################################################
X = data[['X1','X2','X3','X4','X5']].values
regressor = MLPRegressor(hidden_layer_sizes=(10,5,), activation='relu',
                                          solver='adam', alpha=1e-7, batch_size='auto', learning_rate='adaptive',
                                          learning_rate_init=0.001, power_t=0.9, max_iter=10000000, shuffle=True, random_state=None,
                                          tol=1e-4, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                                          early_stopping=False, validation_fraction=0.1, epsilon=1e-08,
                                          n_iter_no_change=10, max_fun=15000)
######################################################################
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = regressor.fit(X_train, Y_train)
SK_y = regressor.predict(X)
SK_resid = y - SK_y
#####################################################
#               MARS
#####################################################
mars = Earth() #max_degree=(1),penalty=(1.0), endspan=(5)
mars.fit(X,y)
print(mars.trace())
print(mars.summary())

mars_y = mars.predict(X)
mars_resid = y-mars_y
######################################
#Real Vs OLS
df = pd.DataFrame({'Actual': y, 'OLS': ols_y})
df1 = df.head(120)
df1.plot(kind='line', figsize=(10, 8), color=["blue","orange"])
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.savefig('sec_ols.eps', transparent= True)
plt.show()
#Real Vs SKLEARN
df = pd.DataFrame({'Actual': y,'SKLEARN': SK_y})
df1 = df.head(120)
df1.plot(kind='line', figsize=(10, 8), color=["blue","green"])
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.savefig('sec_sk.eps', transparent= True)
plt.show()
#Real Vs MARS
df = pd.DataFrame({'Actual': y,  'MARS': mars_y})
df1 = df.head(120)
df1.plot(kind='line', figsize=(10, 8), color=["blue","red"])
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.savefig('sec_mars.eps', transparent= True)
plt.show()
#Real Vs ALL
df = pd.DataFrame({'Actual': y, 'OLS': ols_y, 'SKLEARN': SK_y, 'MARS': mars_y})
df1 = df.head(120)
df1.plot(kind='line', figsize=(10, 8), color=["blue","orange","green", "red"])
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.savefig('sec_all.eps', transparent= True)
plt.show()
#Residuos OLS, SKLEARN, MARS
df = pd.DataFrame({'OLS': ols_resid})
df1 = df.head(120)
df1.plot(kind='hist', figsize=(10, 8), color="orange")
plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.savefig('hist_ols.eps', transparent= True)
#plt.show()
df = pd.DataFrame({'SKLEARN': SK_resid})
df1 = df.head(120)
df1.plot(kind='hist', figsize=(10, 8), color="green")
plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.savefig('hist_sk.eps', transparent= True)
#plt.show()
df = pd.DataFrame({'MARS': mars_resid})
df1 = df.head(120)
df1.plot(kind='hist', figsize=(10, 8), color="red")
plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.savefig('hist_mars.eps', transparent= True)
plt.show()
plt.figure(figsize=(10,8))
plt.hist(ols_resid, 10, alpha=1, color="orange", label = 'OLS')
plt.hist(SK_resid, 10, alpha=1, color="green", label = 'SKLEARN')
plt.hist(mars_resid,10,  alpha=1, color="red", label = 'MARS')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.legend(loc='upper right')
plt.savefig('hist_all.eps', transparent= True)
plt.show()

test = pd.DataFrame.from_dict(
	dict([("OLS", [met1.rsquared_adj, skm.mean_absolute_error(y, ols_y), met1.mse_resid,  np.sqrt(skm.mean_squared_error(y, ols_y))]),
		 ("ANN", [regressor.score(X_test, Y_test), skm.mean_absolute_error(y, SK_y), skm.mean_squared_error(y, SK_y),  np.sqrt(skm.mean_squared_error(y, SK_y))]),
		 ("MARS", [mars.grsq_,skm.mean_absolute_error(y, mars_y), mars.mse_,  np.sqrt(skm.mean_squared_error(y, mars_y))])]),
	orient="index",
	columns=["R2_ajust", "MAE", "MSE",  "RMSE"],
	)
print(test)

###############################
print()

test1 = pd.DataFrame.from_dict(
	dict([("OLS", [sms.jarque_bera(met1.resid)[1], sms.normal_ad(met1.resid)[1],sci.stats.shapiro(ols_resid)[1]]),
		 ("ANN", [sms.jarque_bera(SK_resid)[1], sms.normal_ad(SK_resid)[1], sci.stats.shapiro(SK_resid)[1]] ),
		 ("MARS", [sms.jarque_bera(mars_resid)[1], sms.normal_ad(mars_resid)[1], sci.stats.shapiro(mars_resid)[1]])]),
	orient="index",
	columns=[ "JB", "AD", "SW"],
	)
print(test1)
