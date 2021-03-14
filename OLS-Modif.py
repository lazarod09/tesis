#####################################################
# Import librarys Need
#####################################################
import statsmodels.formula.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats import outliers_influence as oi
import pandas as pd  # Import XLS File
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
#####################################################
# Define Vars
#####################################################
# Load Data
data = pd.read_excel(r'BD.xls', sheet_name='Tulipan')
####################
######################################################################
## Generate models 2^k-1
vardep='Y~'
varind = ['X1','X2','X3','X4','X5']
model=[]
def potencia(c):
    if len(c) == 0:
        return[[]]
    a = potencia(c[:-1])
    return a+[s+[c[-1]] for s in a]
def imprime(c,prefix):
    for e in sorted(c, key=lambda s: (len(s), s)):
        model.append(prefix +" +".join(e))
    return model
models =imprime( potencia(varind), vardep)
del models[0]
######################################################################

# Declaration Vars
####################
mayor1 = 0
mayor2 = 0
mayorCL = 0
m_AIC = 0
m_BIC = 0
sign = 0.05
test_bmodel = []
less_alpha = ['fpvalue', 'CUSUM']
high_alpha = ['LML', 'RB', 'RESET', 'JB', 'AD', 'SW', 'TRESID', 'BPG', 'WH', 'ARCH', 'BG','JK']
cerouno = ['TOL', 'CP']
df = pd.DataFrame(columns=['r_adj','fpvalue','tpvalue', 'AD', 'TRESID', 'BPG','BG','TOL','AIC','CP'])

# #Operations
#####################################################
data_models = []
matriz_dist = []
pos = -1
for i in range(0, len(models)):
    results = sm.ols(formula=models[i], data=data).fit()
    dic_model = {}
    dic_test = {}
    dic_model['model'] = models[i]
    test_ = []
########################################################################################################################
    # Calculo CP-Mallow
    cp_value = ((results.ssr/results.mse_resid))
########################################################################################################################
    # mayor AIC
    if results.aic > m_AIC:
        m_AIC = results.aic
    #mayor BIC
    if results.bic > m_BIC:
        m_BIC = results.bic
########################################################################################################################
    # Calculo VIF & TOL
    vif = 0
    tol = 0
    for h in range(1, len(results.model.exog_names)):
        if oi.variance_inflation_factor(results.model.exog, h) > vif:
            vif = oi.variance_inflation_factor(results.model.exog, h)
        if (1 / vif) > tol:
            tol = (1 / vif)
########################################################################################################################
#   # T Sign
    pv_t = 0
    tpvalues = 0
    for h in range(1, len(results.pvalues)):
        if results.pvalues[h] < sign:
            pv_t += 1
    if pv_t == (len(results.pvalues)-1):
        tpvalues = 1
    else:
        tpvalues = 0
########################################################################################################################
#   ##  TEST ###
    test = [{'name': 'fpvalue', 'test': results.f_pvalue},
            {'name': 'tpvalue', 'test': tpvalues},
            {'name': 'AD', 'test': sms.normal_ad(results.resid)[1]},
            {'name': 'TRESID', 'test': sms.ztest(results.resid,x2=None,value=0, alternative='two-sided', usevar='pooled', ddof=1.0)[1]},
            {'name': 'BPG', 'test': sms.het_breuschpagan(results.resid, results.model.exog)[1]},
            {'name': 'BG', 'test': sms.acorr_breusch_godfrey(results, nlags=2)[1]},
            {'name': 'TOL', 'test': tol},
            {'name': 'r_adj', 'test': results.rsquared_adj}
            ]
    dic_model['test'] = test
    data_models.append(dic_model)
########################################################################################################################
#       Actualizar valores AIC y BIC (Indices) y Guardar solo los valores necesarios
########################################################################################################################
for i in range(0, len(models)):
    dic_model_matriz = {}
    dic_model_matriz['model'] = data_models[i]['model']
    test_matriz_dist = []
    fpvalue = 10
    tpvalue = 10
    CUSUM = 10
    LML = 10
    RB = 10
    RESET = 10
    JB = 10
    AD = 10
    SW = 10
    TRESID = 10
    BPG = 10
    WH = 10
    ARCH = 10
    BG = 10
    JK = 10
    TOL = 10
    VIF = 10
    CP = 10
    r_adj = 10
    for j in range(0, len(data_models[i]['test'])):
        test_ = data_models[i]['test'][j]
        dic_matriz_dist = {}
        dic_matriz_dist['name'] = test_['name']
        value = 0
        z = 0
        #if type(test[j]['test']) is np.float64 or np.float or np.int:
        if (test_['name']) == 'RESET':
            z = str(test_['test']).split(',')
            value = float(str(z[1].split('=')[1]))
            
        elif (test_['name']) == 'JK':
            z = str(str(test_['test']).split(']')[0])
            value = float(z[1:])
        else:
            value = test_['test']
##############################################
#        ### Pruebas de Estructura ####
#        #CUSUM  values
        if test[j]['name'] == 'CUSUM':
            dic_matriz_dist['value'] = []
            if value < sign:
                dic_matriz_dist['value'] = 3
                CUSUM = 3
            else:
                dic_matriz_dist['value'] = 1
                CUSUM = 0

##############################################
#        ### Pruebas de Linealidad ####
#        #LM Values
        if test[j]['name'] == 'LML':
            dic_matriz_dist['value'] = []
            if sign <= value:
                dic_matriz_dist['value'] = 3
                LML = 3
            else:
                dic_matriz_dist['value'] = 1
                LML = 0
#        #RB Values
        if test[j]['name'] == 'RB':
            dic_matriz_dist['value'] = []
            if sign <= value:
                dic_matriz_dist['value'] = 3
                RB = 3
            else:
                dic_matriz_dist['value'] = 1
                RB = 0
#        #RESET Values
        if test[j]['name'] == 'RESET':
            dic_matriz_dist['value'] = []
            if sign <= value:
                dic_matriz_dist['value'] = 3
                RESET = 3
            else:
                dic_matriz_dist['value'] = 1
                RESET = 0
##############################################
#        ### Pruebas de Normalidad ####
#        # AD Values
        if test[j]['name'] == 'AD':
            dic_matriz_dist['value'] = []
            dic_matriz_dist['value'] = []
            if sign <= value:
                dic_matriz_dist['value'] = 3
                AD = 3
            else:
                dic_matriz_dist['value'] = 1
                AD = 0
               
        # if AD == 0 :
        #     continue
##############################################
#        ### Pruebas T Residuos ####
#        # TRESID Values
        if test[j]['name'] == 'TRESID':
            dic_matriz_dist['value'] = []
            if sign <= value:
                dic_matriz_dist['value'] = 3
                TRESID = 3
            else:
                dic_matriz_dist['value'] = 1
                TRESID = 0
##############################################
#        ### Pruebas de Heterocedasticidad ####
#        #BPG Value
        if test[j]['name'] == 'BPG':
            dic_matriz_dist['value'] = []
            if sign <= value:
                dic_matriz_dist['value'] = 3
                BPG = 3
            else:
                dic_matriz_dist['value'] = 1
                BPG = 0

        # if BPG == 0:
        #     continue
##############################################
#        ### Pruebas de Autocorelacion ####
#        #BG Value
        if test[j]['name'] == 'BG':
            dic_matriz_dist['value'] = []
            if sign <= value:
                dic_matriz_dist['value'] = 3
                BG = 3
            else:
                dic_matriz_dist['value'] = 1
                BG = 0
##############################################
#   ## Pruebas de Coherencia Estadistica ####
#        #pvalue F  Global
        if test[j]['name'] == 'fpvalue':
            dic_matriz_dist['value'] = []
            if value < sign:
                dic_matriz_dist['value'] = 3
                fpvalue=3
            else:
                dic_matriz_dist['value'] = 1
                fpvalue=0
#        #All t Sign
        if test[j]['name'] == 'tpvalue':
            dic_matriz_dist['value'] = []
            if value == 1:
                dic_matriz_dist['value'] = 3
                tpvalue = 3
            else:
                dic_matriz_dist['value'] = 1
                tpvalue = 0
##############################################
#        ### Criterios de informacion ####
#        #TOL
        if test[j]['name'] == 'TOL':
            dic_matriz_dist['value'] = []
            if 0.9 <= value:
                dic_matriz_dist['value'] = 3
                TOL = 3
            elif 0.1 < value < 0.9:
                dic_matriz_dist['value'] = 2
                TOL = 2
            else:
                dic_matriz_dist['value'] = 1
                TOL = 0
#        #AIC
        if test[j]['name'] == 'AIC':
            dic_matriz_dist['value'] = []
            if 33 <= value:
                dic_matriz_dist['value'] = 3
                AIC = 3
            elif 33 < value < 66:
                dic_matriz_dist['value'] = 2
                AIC = 2
            else:
                dic_matriz_dist['value'] = 1
                AIC = 0
        # BIC
        if test[j]['name'] == 'BIC':
            dic_matriz_dist['value'] = []
            if value < 33:
                dic_matriz_dist['value'] = 3
                BIC = 3
            elif 33 < value < 66:
                dic_matriz_dist['value'] = 2
                BIC = 2
            else:
                dic_matriz_dist['value'] = 1
                BIC = 0
        # CP Mallow
        if test[j]['name'] == 'CP':
            dic_matriz_dist['value'] = []
            if 0.9 <= value:
                dic_matriz_dist['value'] = 3
                CP = 3
            elif 0.7 < value < 0.9:
                dic_matriz_dist['value'] = 2
                CP = 2
            else:
                dic_matriz_dist['value'] = 1
                CP = 1
        # R Ajustado
        if test[j]['name'] == 'r_adj':
            dic_matriz_dist['value'] = []
            r_adj = value
            if 0.9 <= value:
                dic_matriz_dist['value'] = 3
            elif 70 < value < 0.9:
                dic_matriz_dist['value'] = 2
            else:
                dic_matriz_dist['value'] = 1

##############################################
        test_matriz_dist.append(dic_matriz_dist)  
        test_df = pd.DataFrame.from_dict(
        	dict([(data_models[i]['model'],[r_adj,fpvalue,tpvalue,AD, TRESID, BPG, BG,TOL,AIC, CP])]),
        	orient="index",
        	columns=['r_adj','fpvalue','tpvalue', 'AD', 'TRESID', 'BPG','BG','TOL','AIC','CP'],
        	)
    if (AD == 0) or ( BPG == 0) or ( TOL < 3) or (tpvalue == 0) or (r_adj  < 0.6):
        continue
    df = df.append(test_df)
    dic_model_matriz['test'] = test_matriz_dist
    matriz_dist.append(dic_model_matriz)
    
print(df)
###############################
###############################
print()
ols_y= results.predict()
ols_resid = results.resid
df = pd.DataFrame({'OLS': ols_resid})
df1 = df.head(120)
df1.plot(kind='bar', figsize=(10, 8), color="orange")
plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.savefig('hist_ols.eps', transparent= True)