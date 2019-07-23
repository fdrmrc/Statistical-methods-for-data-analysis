import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error


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


#generate linear regression model on train set
reg=linear_model.Ridge(alpha=1.0)
reg.fit(predictorTrain_std,lpsaTrain)

print('Ridge regression coefficients are:', reg.coef_)
print('Ridge regression intercept is:', reg.intercept_)
print('Ridge regression R^2 is:', reg.score(predictorTrain_std,lpsaTrain))

print('\n')
#plot Ridge coefficients as a function of the regularization parameters

alphas=np.logspace(-1,4,200) #vector of regularization parameters. See documentation for logspace (goes from 1e-1 to 1e4 with 200 sample points)
coefs=list()
for i in range(len(alphas)):
    reg=linear_model.Ridge(alpha=alphas[i])
    reg.fit(predictorTrain_std,lpsaTrain)
    coefs.append(reg.coef_)

#plot alphas against the related coefficients: cfr http://www.ds100.org/sp19/assets/lectures/regularization.pdf#Navigation24
plt.plot(alphas,coefs)
plt.xlabel('regularization parameter')
plt.ylabel('coefficients')
#plt.legend('lcavol','lweight','age','lbph','svi','lcp','gleason','pgg45') save coeff in a matrix to plot also the legend
plt.title('Regularization parameter against Ridge coeffs')
plt.show()


#find the best regularization parameter by cross-validation
reg = linear_model.RidgeCV(alphas=np.logspace(-1,4,200),store_cv_values=True)
reg.fit(predictorTrain_std,lpsaTrain)
print('Dimensions of cv_values: ', reg.cv_values_.shape)
print('best alpha: ', reg.alpha_)

#Compute the mean by column of cv_values_: I obtain the average cross-validation error for each alpha. This is the value that has to be minimized 
colMean=np.zeros([reg.cv_values_.shape[1],1])
for i in range(reg.cv_values_.shape[1]):
    colMean[i]=reg.cv_values_[:,i].mean()

#Plot the leave-one-out cross validation error
plt.plot(alphas,colMean)
plt.xlabel('reg. parameters')
plt.ylabel('Average cross validation error')
plt.show()

#Identify the minimum leave-one-out cross-validation error and the related alpha, model coefficients and performance
minAv=np.min(colMean)
minIndex=np.where(colMean==minAv) #tell me where colMean has its minimum
best_alpha=alphas[int(minIndex[0])] #casted in order to pass the index
reg=linear_model.Ridge(alpha=best_alpha)
reg.fit(predictorTrain_std,lpsaTrain)
print('best_alpha by hand: ', best_alpha)

print('New Ridge regression coefficients are:', reg.coef_)
print('New Ridge regression intercept is:', reg.intercept_)
print('New Ridge regression R^2 is:', reg.score(predictorTrain_std,lpsaTrain))


print('\n')
#Identify the minimum 10-folds cross-validation error and the related alpha, model coefficients and performance

reg = linear_model.RidgeCV(alphas=np.logspace(-1,4,200),cv=10)
reg.fit(predictorTrain_std,lpsaTrain)
print('best_alpha: ',reg.alpha_)
print('coefficients: ', reg.coef_)
print('R^2: ',reg.score(predictorTrain_std,lpsaTrain))

print('\n')


#LASSO regression on train set
reg=linear_model.Lasso(alpha=0.1)
reg.fit(predictorTrain_std,lpsaTrain)
print('Lasso regression coefficients are:', reg.coef_)
print('Lasso regression intercept is:', reg.intercept_)
print('Lasso regression R^2 is:', reg.score(predictorTrain_std,lpsaTrain))

#Plot Lasso coefficients as a function of the regularization parameter
alphas_lasso, coefs_lasso, _=linear_model.lasso_path(predictorTrain_std,lpsaTrain)

for k in range(len(reg.coef_)):
    plt.plot(alphas_lasso,coefs_lasso[k,:],linestyle='--')
    #plt.legend()

plt.xlabel('alpha')
plt.ylabel('coefficients')
plt.title('Lasso paths')
plt.show()


#Find the best regularization parameter by cross-validation

model = linear_model.LassoCV(cv=20).fit(predictorTrain_std, lpsaTrain)
m_log_alphas = -np.log10(model.alphas_)

plt.figure
plt.plot(m_log_alphas, model.mse_path_, ':') #mean square error
plt.title('Mean square error for each fold')
plt.plot(m_log_alphas, model.mse_path_.mean(axis=1), 'k', #average across all folds
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha: CV estimate') #show best regularization parameter in the plot 

plt.xlabel('-log(alpha)')
plt.ylabel('Mean square error')
plt.legend()
plt.show()

#Show the best alpha, model coefficients and performance on training  set
print('\n')
print('The best alpha is: ',model.alpha_)
print('Model coefficients are: ',model.coef_)
print('Model R^2: ',model.score(predictorTrain_std,lpsaTrain))