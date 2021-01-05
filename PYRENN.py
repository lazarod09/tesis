import pandas as pd
import matplotlib.pyplot as plt
import pyrenn as prn
import numpy as np
from sklearn.model_selection import train_test_split
######################################################################
data = pd.read_csv(r'yeni.csv')
Y = data['Y'].values
X = data[['X1', 'X2', 'X3', 'X4', 'X5']].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

Y = np.array([data['Y'].values])
X = np.transpose(X)
#X = np.array([data['X1'].values,data['X2'].values,data['X3'].values,data['X4'].values,data['X5'].values])
#X_test = np.array([data['X1test'].values,data['X2test'].values,data['X3test'].values,data['X4test'].values,data['X4test'].values])
#Y_test = np.array([data['Y1test'].values])
X_test = np.transpose(X_test)
X1= np.array([data['X1'].values])
######################################################################

######################################################################

net = prn.CreateNN([5,10,5,1],dIn=[0],dIntern=[],dOut=[1])

net = prn.train_LM(X,Y,net,verbose=True,k_max=100,E_stop=1e-5)


y = prn.NNOut(X,net)
ytest = prn.NNOut(X_test,net)

y = np.array([y])

######################################################################
fig = plt.figure(figsize=(11,7))
ax0 = fig.add_subplot(211)
ax1 = fig.add_subplot(212)
fs=18

#Train Data
ax0.set_title('Train Data',fontsize=fs)
ax0.plot(X1,y,color='b',lw=2,label='NN Output')
ax0.plot(X1,Y,color='r',marker='None',linestyle=':',lw=3,markersize=8,label='Train Data')
ax0.tick_params(labelsize=fs-2)
ax0.legend(fontsize=fs-2,loc='upper left')
ax0.grid()

#Test Data
ax1.set_title('Test Data',fontsize=fs)
ax1.plot(Y_test,ytest,color='b',lw=2,label='NN Output')
ax1.plot(Y_test,ytest,color='r',marker='None',linestyle=':',lw=3,markersize=8,label='Test Data')
ax1.tick_params(labelsize=fs-2)
ax1.legend(fontsize=fs-2,loc='upper left')
ax1.grid()

fig.tight_layout()
plt.show()