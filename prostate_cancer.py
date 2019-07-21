import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data=pd.read_csv('prostate.csv',sep=',',engine='python',index_col='train')

print('dimensioni dei dati: \n',data.shape)
print(data.describe()) #show basic statistics


#generazione del train set: devo identificare le righe che corrispondono a train=T
dataTrain=data.loc['T'] #estrae le righe che corrispondono al train set, prendendole dal data frame.
print('La dimensione del train set è: \n',data.shape)

#Ora rimuovo ciò che non serve, ossia la colonna lpsa, che è ciò che devo utilizzare come y nella regressione, e la salvo.
lpsaTrain=dataTrain.iloc[:,-1] #prendo l'ultimo
dataTrain=dataTrain.drop('lpsa',axis=1)
predictorTrain=dataTrain

#Genero il test set.
dataTest=data.loc['F']
lpsaTest=dataTest.iloc[:,-1]
dataTest=dataTest.drop('lpsa',axis=1)

#matrice di correlazione
M=dataTrain.corr() 
print(M)


#standardizzare le variabili
#1. calcolo la media per ogni colonna
predictorTrainMean=np.zeros([1,np.array(dataTrain.shape)[1]])
for i in range (0,np.array(dataTrain.shape)[1]):
    predictorTrainMean[0,i]=predictorTrain.iloc[:,i].mean()

#2. calcolo standard deviation
predictorTrainStd=np.zeros([1,np.array(dataTrain.shape)[1]])
for i in range (0,np.array(dataTrain.shape)[1]):
    predictorTrainStd[0,i]=predictorTrain.iloc[:,i].std()

#3. standardizing variables

for i in range (0,np.array(dataTrain.shape)[1]):
    predictorTrain.iloc[:,i]=(predictorTrain.iloc[:,i]-predictorTrainMean[0,i])/(predictorTrainStd[0,i])

predictorTrain_std=predictorTrain #assegno il train set standardizzato a una nuova variaibile

#Some histograms

plt.subplot(221)
predictorTrain_std.iloc[:,0].hist()
plt.title('lcavol')

plt.subplot(222)
predictorTrain_std.iloc[:,1].hist()
plt.title('lweight')

plt.subplot(223)
predictorTrain_std.iloc[:,2].hist()
plt.title('age')

plt.subplot(224)
predictorTrain_std.iloc[:,-2].hist()
plt.title('gleason')
plt.show()


#Linear regression
reg=LinearRegression(normalize=True).fit(predictorTrain_std,lpsaTrain)
print('coefficienti: \n',reg.coef_) #coerenti con quanto visto sul libro
print('intercept: \n',reg.intercept_)
print('determination coefficient: \n',reg.score(predictorTrain_std,lpsaTrain))


#Using statsmodel regressione
mod=sm.OLS(lpsaTrain,predictorTrain_std)
res=mod.fit()
print(res.summary())

#I can drop age, lcp, gleason, pgg45 by looking at the Z-score


# F statistics
#1. Residual sum of squares for model without drops
RSS_no_drop=np.sum((lpsaTrain-reg.predict(predictorTrain_std))**2)

#2. Residual sum of squares for model WITH drops

predictorTrain_std=predictorTrain_std.drop('age',axis=1)
predictorTrain_std=predictorTrain_std.drop('lcp',axis=1)
predictorTrain_std=predictorTrain_std.drop('gleason',axis=1)
predictorTrain_std=predictorTrain_std.drop('pgg45',axis=1)

reg=LinearRegression(normalize=True).fit(predictorTrain_std,lpsaTrain)

RSS_with_drop=np.sum((lpsaTrain-reg.predict(predictorTrain_std))**2)

#Compute F-statistics
F=((RSS_with_drop-RSS_no_drop)/(9-5))/(RSS_no_drop/(67-1-8))
print('F-statistics is: \n',F)

#H_0= model without age,lcp,gleason,pgg45 is correct
#Since p-value is 0.17>0.05, the H_0 hyp is NOT REJECTED