import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import itertools 

data=pd.read_csv('prostate.csv',sep=',',engine='python',index_col='train')

#generazione del train set: devo identificare le righe che corrispondono a train=T
dataTrain=data.loc['T'] #estrae le righe che corrispondono al train set, prendendole dal data frame.
#print('La dimensione del train set è: \n',data.shape)

#Ora rimuovo ciò che non serve, ossia la colonna lpsa, che è ciò che devo utilizzare come y nella regressione, e la salvo.
lpsaTrain=dataTrain.iloc[:,-1] #prendo l'ultimo
dataTrain=dataTrain.drop('lpsa',axis=1)
predictorTrain=dataTrain

#Genero il test set.
dataTest=data.loc['F']
lpsaTest=dataTest.iloc[:,-1]
dataTest=dataTest.drop('lpsa',axis=1)


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



#BEST SUBSET SELECTION

results=pd.DataFrame(columns=['num_features', 'features', 'MAE','RSS','R^2'])

for k in range (1,dataTrain.shape[1]+1):
    for subset in itertools.combinations(range(dataTrain.shape[1]),k):
        subset=list(subset)
        linreg=LinearRegression(normalize=True).fit(predictorTrain_std.iloc[:,subset],lpsaTrain)
        linreg_pred=linreg.predict(dataTrain.iloc[:,subset]) #prediction using train set
        linreg_mae = np.mean(np.abs(lpsaTrain - linreg_pred))
        RSS=np.sum((lpsaTrain -linreg_pred)**2)
        Rsquare=linreg.score(predictorTrain_std.iloc[:,subset] ,lpsaTrain)
        results = results.append(pd.DataFrame([{'num_features': k,
                                                'features': subset,
                                                'MAE': linreg_mae,
                                                'RSS':RSS,
                                                'R^2':Rsquare}]),sort=True)

#sort values
        
results_sort=results.sort_values('MAE')

results_sort_RSS=results.sort_values('RSS') #equals to the sort above

best_subset_model=LinearRegression(normalize=True).fit(dataTrain.iloc[:,results_sort['features'].iloc[0]],lpsaTrain) #fit(X_best,y_train)
best_subset_coefs=best_subset_model.coef_

print('Best subset Regression RSS : {}'.format(np.round(results_sort['RSS'].iloc[0],3)))
print('Best subset Regression R^2 : {}'.format(np.round(results_sort['R^2'].iloc[0],3)))


#Plot of subset size vs. RSS
results['min_RSS'] = results.groupby('num_features')['RSS'].transform(min)
plt.scatter(results.iloc[:,4],results.iloc[:,1],alpha=0.8,color='grey',s=5) #x:subset size, y:RSS
plt.xlabel('k: subset size')
plt.ylabel('RSS for every model')
plt.title('RSS - Best subset selection')
plt.legend()
plt.plot(results.iloc[:,4],results.iloc[:,-1],color='red',linestyle=':')

plt.show()

#Plot of subset size vs. R^2
results['max_R^2']=results.groupby('num_features')['R^2'].transform(max)
plt.scatter(results.iloc[:,4],results.iloc[:,2],alpha=0.8,color='darkblue',s=5) #x:subset size, y:R^2
plt.xlabel('k: subset size')
plt.ylabel('R^2 for every model')
plt.title('R^2 - Best subset selection')
plt.legend()
plt.plot(results.iloc[:,4],results.iloc[:,-1],color='green',linestyle=':')

plt.show()