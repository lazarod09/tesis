# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pyearth import Earth
import matplotlib.pyplot as plt
from sklearn import metrics
import statistics as stat
import statsmodels.stats.api as sms
import scipy as sci

data = pd.read_excel(r'BD.xlsx', sheet_name='Cfgos', engine='openpyxl')
y = data['Y'].values
X = data[['X1', 'X2','X4', 'X5']].values

#Fit an Earth model
model = Earth() #max_degree=(1),penalty=(1.0), endspan=(5)
model.fit(X,y)

#Print the model
print(model.trace())
print(model.summary())
print(np.sqrt(model.mse_))
print(model.rsq_)
#Plot the model
y_hat = model.predict(X)
resid = y-y_hat
# pyplot.figure()
# pyplot.plot(X[:,6],y,'r.')
# pyplot.plot(X[:,6],y_hat,'b.')
# pyplot.xlabel('x_6')
# pyplot.ylabel('y')
# pyplot.title('Simple Earth Example')
# pyplot.show()
df = pd.DataFrame({'Actual': y})
df1 = df.head(25)
df1.plot(kind='box', figsize=(10, 8)) 
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
df = pd.DataFrame({'Actual': y, 'Predicted': y_hat})
df1 = df.head(120)
df1.plot(kind='line', figsize=(10, 8)) 
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
df = pd.DataFrame({'Residuos': resid})
df1 = df.head(120)
df1.plot(kind='hist', figsize=(10, 8)) 
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
df1 = df.head(50)
df1.plot(kind='bar', figsize=(10, 8)) 
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y, y_hat))
print('Mean Squared Error:', metrics.mean_squared_error(y, y_hat))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y, y_hat)))
print('Normalidad')
print('AD(pvalue)', sms.normal_ad(resid)[1])
print('JB', sms.jarque_bera(resid)[1])
print('S-W', sci.stats.shapiro(resid)[1])
print('lilliefors', sms.lilliefors(resid)[1])
print('Descriptivo de residups y ttest')
print('mu(resid)=0 test', sms.ztest(resid,x2=None,value=0, alternative='two-sided', usevar='pooled', ddof=1.0)[1])
print('mu(Residuos)',stat.mean(resid), 'dv(Residuos)', np.sqrt(stat.variance(resid)))
