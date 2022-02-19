import numpy as np
from numpy.core.fromnumeric import mean
from sklearn.svm import SVR
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf  

rate = pd.read_excel("fullrate21294.xlsx",sheet_name="rate")
market = pd.read_excel("fullrate21294.xlsx",sheet_name="maket")
coef = pd.read_excel("fullrate21294.xlsx",sheet_name="betas")
stock = rate.iloc[:,:].values
bt = coef.iloc[60:95,].values
var = market.iloc[:,3].values
rf = market.iloc[:,1].values

model = SVR(kernel="linear",C=0.001)
x_real = stock[60:95,]
var_real = var[60:95,]
RF = rf[60:95,]
output = np.zeros((34,212))
for j in range(212):
    for i in range(34):
        x  = rate.iloc[:(i+61),j].values
        x = x.reshape(len(x),1)
        x1 = x[:len(x)-1,]
        x2 = x[len(x)-1:len(x),]
        y = market.iloc[:(i+60),3].values
        m = model.fit(x1,y)
        output[i,j] = m.predict(x2)

# RMSE
print("RMSE: ",np.sqrt(mean((x_real-output)**2)))
# MsE
print("MSE: ",mean((x_real-output)**2))
# MAE
print("MAE: ",mean(abs(x_real-output)))


# stage 2
RMSESVR = np.zeros((212,1))
RMSECAPM = np.zeros((212,1))
VAR = np.zeros((212,1))
SD = np.zeros((212,1))
MEAN = np.zeros((212,1))
BETA = np.zeros((212,1))
for i in range(212):
    SD[i] = np.std(x_real[:,i])
    MEAN[i] = np.mean(x_real[:,i])
    RMSESVR[i] = np.sqrt(mean((output[:,i]-x_real[:,i])**2))
    RMSECAPM[i] = np.sqrt(mean((RF+bt[:,i]*var_real-x_real[:,i])**2))
    df = pd.concat([rate.iloc[60:95:,i],market.iloc[60:95,3]],axis=1)
    df = df.set_axis(["x",'y'],axis=1)
    reg = smf.ols("y~x",data=df).fit()
    pre = reg.params[0]+reg.params[1]*rate.iloc[60:95,i]
    resid = rate.iloc[60:95,i] - pre
    BETA[i] = reg.params[1]
    VAR[i] = np.var(resid)
datasvr =np.concatenate((RMSESVR,RMSECAPM,VAR,SD,MEAN,BETA),axis=1)
dta = pd.DataFrame(datasvr)
dta = dta.set_axis(['RMSESVR','RMSECAPM','VAR','SD','MEAN','BETA'],axis=1)
modelerror = smf.ols("RMSESVR~RMSECAPM+VAR+SD+MEAN+BETA",data=dta).fit()
print(modelerror.summary())

# Visualization 
dta["BETA"].plot(kind = "hist",bins = 15)
plt.show()
dta.plot(kind = "scatter",x="BETA",y="MEAN")
plt.show()
print(stock.mean(axis=0))
print(rate["AAM"].describe())
print(dta.describe())
rate["AAM"].plot(kind = "box").plot()
plt.show()
Date = pd.to_datetime(market["day"])
plt.plot(Date,rate["AAM"])
plt.show()
















