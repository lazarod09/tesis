#####################################################
# Import librarys Need
#####################################################
import statsmodels.formula.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats import outliers_influence as oi
import pandas as pd  # Import XLS File
import numpy as np
import scipy as sci

#####################################################
# Define Vars
#####################################################
# Load Data
#data = pd.read_csv(r'yeni.csv')
data = pd.read_excel(r'BD.xlsx', sheet_name='Cfgos')
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
cerouno = ['TOL', 'CP', 'r_adj']
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
    cp_value = ((results.ssr/results.mse_resid)-len(results.resid)+2*results.df_model)/ results.df_model
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
            {'name': 'CUSUM', 'test': sms.breaks_cusumolsresid(results.resid)[1]},
            {'name': 'LML', 'test': sms.linear_lm(results.resid, results.model.exog)[1]},
            {'name': 'RB', 'test': sms.linear_rainbow(results)[1]},
            {'name': 'RESET', 'test': oi.reset_ramsey(results)},
            {'name': 'JB', 'test': sms.jarque_bera(results.resid)[1]},
            {'name': 'AD', 'test': sms.normal_ad(results.resid)[1]},
            {'name': 'SW', 'test': sci.stats.shapiro(results.resid)[1]},
            {'name': 'TRESID', 'test': sms.ztest(results.resid,x2=None,value=0, alternative='two-sided', usevar='pooled', ddof=1.0)[1]},
            {'name': 'BPG', 'test': sms.het_breuschpagan(results.resid, results.model.exog)[1]},
            {'name': 'WH', 'test': sms.het_white(results.resid, results.model.exog)[1]},
            {'name': 'ARCH', 'test': sms.het_arch(results.resid)[1]},
            {'name': 'BG', 'test': sms.acorr_breusch_godfrey(results, nlags=2)[1]},
            {'name': 'JK', 'test': sms.acorr_ljungbox(results.resid, lags=[10])[1]},
            {'name': 'VIF', 'test': vif},
            {'name': 'TOL', 'test': tol},
            {'name': 'AIC', 'test': abs(results.aic)},
            {'name': 'BIC', 'test': abs(results.bic)},
            {'name': 'CP', 'test': abs(cp_value)},
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
#        # Matriz Dist
##############################################
#   ## Pruebas < sign ####
        if test_['name'] in less_alpha:
            if value < sign:
                dic_matriz_dist['value'] = 3
            else:
                dic_matriz_dist['value'] = 1
##############################################
#   ## Pruebas   ####
        if test_['name'] in high_alpha:
            if sign <= value :
                dic_matriz_dist['value'] = 3
            else:
                dic_matriz_dist['value'] = 1
##############################################
#        #All t Sign
        if test_['name'] == 'tpvalue':
            if value == 1:
                dic_matriz_dist['value'] = 3
            else:
                dic_matriz_dist['value'] = 1
##############################################
# Entre 0 y 1
        if test_['name'] in cerouno:
            if 0.9 <= value:
                dic_matriz_dist['value'] = 3
            elif 0.7 < value < 0.9:
                dic_matriz_dist['value'] = 2
            else:
                dic_matriz_dist['value'] = 1
#        ### Criterios de informacion ####
#        #VIF
        if test_['name'] == 'VIF':
            if value < 5:
                dic_matriz_dist['value'] = 3
            elif 5 < value < 10:
                dic_matriz_dist['value'] = 2
            else:
                dic_matriz_dist['value'] = 1
#        #AIC
        if test_['name'] == 'AIC':
            if value/m_AIC <= 0.33:
                dic_matriz_dist['value'] = 3
            elif 0.33 < value/m_AIC < 0.66:
                dic_matriz_dist['value'] = 2
            else:
                dic_matriz_dist['value'] = 1
        # BIC
        if test_['name'] == 'BIC':
            dic_matriz_dist['value'] = []
            if value/m_BIC <= 0.33:
                dic_matriz_dist['value'] = 3
            elif 0.33 < value/m_BIC < 0.66:
                dic_matriz_dist['value'] = 2
            else:
                dic_matriz_dist['value'] = 1
##############################################
        test_matriz_dist.append(dic_matriz_dist)
    dic_model_matriz['test'] = test_matriz_dist
    matriz_dist.append(dic_model_matriz)
#print('############DATA MODELS##############')
#print(data_models)
#print('############MATRIZ DISTANCE##############')
print(matriz_dist)
#print('############MATRIZ DISTANCE##############')
mayores = [object for x in range(len(test))]
for j in range(0, len(test)):
   mayor = 0
   dic = {}
   dic['name'] = test[j]['name']
   for i in range(0, len(models)):
       name = matriz_dist[i]['test'][j]['name']
       if matriz_dist[i]['test'][j]['value'] > mayor:
           mayor = matriz_dist[i]['test'][j]['value']
           dic['value'] = mayor
   mayores[j] = dic
#print('#####################MAYORES###########################')
#print(mayores)
matriz_norm = matriz_dist
for j in range(0, len(test)):
   for i in range(0, len(models)):
       #name = matriz_dist[i]['test'][j]['name']
       a = mayores[j]
       value = a['value']
       value2 = matriz_dist[i]['test'][j]['value']
       new_value = (value - value2) / 2
       matriz_norm[i]['test'][j]['value'] = new_value
print('###############MATRIZ NORM######################')
print(matriz_norm)

criterios = [
   {'name': 'CoherenciaEst', 'test': ['fpvalue', 'tpvalue'], 'value':0, 'supremo': 0},
   {'name': 'Estabilidad', 'test': ['CUSUM'], 'value': 0, 'supremo': 0},
   {'name': 'Lineabilidad', 'test': ['LML', 'RB', 'RESET'], 'value': 0, 'supremo': 0},
   {'name': 'Normalidad', 'test': ['JB', 'AD','SW'], 'value':0, 'supremo': 0},
   {'name': 'tResiduos', 'test': ['TRESID'], 'value':0, 'supremo': 0},
   {'name': 'Heterocedasticidad', 'test': ['BPG', 'WH', 'ARCH'], 'value': 0, 'supremo': 0},
   {'name': 'Autocorrelacion', 'test': ['BG', 'JK'], 'value': 0, 'supremo': 0},
   {'name': 'Multicolinealidad', 'test': ['VIF', 'TOL'], 'value': 0, 'supremo': 0},
   {'name': 'CriteriosInf', 'test': ['AIC', 'BIC', 'CP'], 'value': 0, 'supremo': 0},
   {'name': 'BondadAjuste', 'test': ['r_adj'], 'value': 0, 'supremo': 0},
]
################################################################################################################
##SI criterio de normalidad , Heterocedasticidad dan 2 mal eliminar ese modelo (si es mayor que )
# Si criterio Multicolinealidad da 1 mal eliminar ese modelo

matriz_criterios = []
for i in range(0, len(models)):
   dic_criterios_modelo = {}
   dic_criterios_modelo['name'] = matriz_dist[i]['model']
   current_criterios = []
   suma_modelo = 0
   mayor = 0
   for h in range(0, len(criterios)):
       suma_criterio = 0
       dic_criterio = {}
       dic_criterio['name'] = criterios[h]['name']
       test_criterios = criterios[h]['test']
       for j in range(0, len(test)):
           #value = 0
           name = matriz_dist[i]['test'][j]['name']
           value = matriz_dist[i]['test'][j]['value']
           if name in test_criterios:
               suma_criterio += value
           if name == 'Normalidad' and suma_criterio>1:
               suma_criterio += 20
           if name == 'Heterocedasticidad' and suma_criterio>1:
               suma_criterio += 20
           if suma_criterio > mayor:
               mayor = suma_criterio
       suma_modelo += suma_criterio
       dic_criterio['suma'] = suma_criterio
       current_criterios.append(dic_criterio)
   dic_criterios_modelo['criterios'] = current_criterios
   dic_criterios_modelo['suma'] = suma_modelo
   dic_criterios_modelo['mayor'] = mayor
   matriz_criterios.append(dic_criterios_modelo)
################################################################################################################
#print('###############CRITERIOS######################')
#print(matriz_criterios)

minimo_suma = matriz_criterios[0]['suma']
pos_min_suma = -1
minimo_mayor = matriz_criterios[0]['mayor']
pos_min_mayor = -1
for i in range(0, len(models)):
   if matriz_criterios[i]['suma'] < minimo_suma:
       pos_min_suma = i
       minimo_suma = matriz_criterios[i]['suma']
   if matriz_criterios[i]['mayor'] < minimo_mayor:
       pos_min_mayor = i
       minimo_mayor = matriz_criterios[i]['mayor']

equal_models_minimo_norm = []
equal_models_mayor_norm = []
for i in range(0, len(models)):
   if matriz_criterios[i]['suma'] == minimo_suma:
       equal_models_minimo_norm.append(matriz_criterios[i]['name'])
   if matriz_criterios[i]['mayor'] == minimo_mayor:
       equal_models_mayor_norm.append(matriz_criterios[i]['name'])


print('###############MINIMO SUMA ######################')
print(equal_models_minimo_norm)
print('###############MINIMO MAYOR######################')
print(equal_models_mayor_norm)
print('###############Interseccion######################')
print(list(set(equal_models_minimo_norm).intersection(equal_models_mayor_norm)))
print('###############Mejor Modelo######################')
#####################################################
#   Test of Best Models
best_model = list(set(equal_models_minimo_norm).intersection(equal_models_mayor_norm))
for item in best_model:
    for model in data_models:
        if item == model['model']:
            test_bmodel.extend(model['test'])

print(test_bmodel)
#####################################################
#####################################################