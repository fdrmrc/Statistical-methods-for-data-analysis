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
        linreg_pred=linreg.predict(predictorTrain_std.iloc[:,subset]) #prediction using train set
        linreg_mae = np.mean(np.abs(lpsaTrain - linreg_pred))
        RSS=np.sum((lpsaTrain -linreg_pred)**2)
        Rsquare=linreg.score(predictorTrain_std.iloc[:,subset] ,lpsaTrain)
        results = results.append(pd.DataFrame([{'num_features': k,
                                                'features': subset,
                                                'MAE': linreg_mae,
                                                'RSS':RSS,
                                                'R^2':Rsquare}]))

#sort values

#results_sort=results.sort_values('MAE')

results_sort=results.sort_values('RSS')

best_subset_model=LinearRegression(normalize=True).fit(predictorTrain_std.iloc[:,results_sort['features'].iloc[0]],lpsaTrain) #fit(X_best,y_train)
best_subset_coefs=best_subset_model.coef_

print('Best subset Selection RSS : {}'.format(np.round(results_sort['RSS'].iloc[0],3)))
print('Best subset Selection R^2 : {}'.format(np.round(results_sort['R^2'].iloc[0],3)))


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


#FORWARD STEPWISE SELECTION:
#Forward Stepwise begins with a model containing no predictors, and then adds predictors to the model, one at the time.
#At each step, the variable that gives the greatest additional improvement to the fit is added to the model.

remaining_features=list(dataTrain.columns.values) #features to be included
features=[] #start with empty model
features_list=dict() #copy here features used
RSS_list=[]
RSquare_list=[]
for i in range (1,dataTrain.shape[1]+1):
    best_RSS=np.inf

    for combo in itertools.combinations(remaining_features,1):
        reg=LinearRegression().fit(predictorTrain_std[list(combo) + features],lpsaTrain)
        pred=reg.predict(predictorTrain_std[list(combo) + features]) #prediction to compute RSS
        RSS=np.sum((lpsaTrain -pred)**2)
        Rsquare=reg.score(predictorTrain_std[list(combo) + features],lpsaTrain)

        if RSS<best_RSS:
            best_RSS=RSS
            best_feature=combo[0] #choose current combo
            best_Rsquare=Rsquare

    features.append(best_feature)
    remaining_features.remove(best_feature)

    #save for the plot
    RSS_list.append(best_RSS)
    RSquare_list.append(best_Rsquare)
    features_list[i]=features.copy()

print('Forward stepwise selection RSS : {}'.format(np.round(best_RSS,3)))
print('Forward stepwise selection R^2 : {}'.format(np.round(best_Rsquare,3)))

#plot: x=subset size, y=RSS

plt.scatter(np.arange(1,dataTrain.shape[1] + 1),RSS_list)
plt.xlabel('k: subset size')
plt.ylabel('RSS')
plt.legend(' RSS ')
plt.title('RSS - Forward stepwise selection')
plt.show()


#BACKWARD STEPWISE SELECTION

#Initialization variables
Y = lpsaTrain
X = predictorTrain_std
k = X.shape[1]

features = list(X.columns.values)
reg = LinearRegression().fit(X[features],Y)
pred=reg.predict(X[features])
RSS=np.sum((Y -pred)**2)
RSS_list, R_squared_list = [RSS], [] #Due to 1 indexing of the loop...

for i in range(1,k):
    worst_RSS = -np.inf

    for combo in itertools.combinations(features,1):
            features.remove(combo[0])
            reg = LinearRegression().fit(X[features],Y)   #Store temp result
            pred=reg.predict(X[features])
            RSS=np.sum((Y -pred)**2)
            if RSS > worst_RSS:
                worst_feature = combo[0] #Combo è una lista di 1 elemento e quindi per accedere a quello devi fare combo[0]
                            #se fai solo combo non riesce a capire che è una stringa ma la vede come una lista
            features.append(combo[0])
    #Updating variables for next loop
    features.remove(worst_feature)
    reg = LinearRegression().fit(X[features],Y)
    pred=reg.predict(X[features])
    RSS=np.sum((Y -pred)**2)
    print(features)
    #Saving values for plotting
    RSS_list.append(RSS)
    #R_squared_list.append(RSS)
x=np.arange(k,0,-1)
print(len(x))
plt.plot(x,RSS_list[0:],'r-o')
plt.title('Plot comparing number of used variables and related RSS')
plt.xlabel('Subset size')
plt.ylabel('RSS')
plt.show()

