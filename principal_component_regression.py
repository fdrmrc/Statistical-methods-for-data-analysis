import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans

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
predictorTest=dataTest.drop('lpsa',axis=1)


#standardizzare le variabili

##1. calcolo la media per ogni colonna
#predictorTrainMean=np.zeros([1,np.array(dataTrain.shape)[1]])
#for i in range (0,np.array(dataTrain.shape)[1]):
#    predictorTrainMean[0,i]=predictorTrain.iloc[:,i].mean()
#
##2. calcolo standard deviation
#predictorTrainStd=np.zeros([1,np.array(dataTrain.shape)[1]])     #avoid loops
#for i in range (0,np.array(dataTrain.shape)[1]):
#    predictorTrainStd[0,i]=predictorTrain.iloc[:,i].std()
#
##3. standardizing variables
#
#for i in range (0,np.array(dataTrain.shape)[1]):
#    predictorTrain.iloc[:,i]=(predictorTrain.iloc[:,i]-predictorTrainMean[0,i])/(predictorTrainStd[0,i])
#
#predictorTrain_std=predictorTrain #assegno il train set standardizzato a una nuova variaibile

predictorTrainMean=predictorTrain.mean()
predictorTrainStd=predictorTrain.std()
predictorTrain_std=(predictorTrain - predictorTrainMean)/(predictorTrainStd) #in this way I avoid the loop on the columns by using methods

print(predictorTrain_std)

# Standardize test set

predictorTestMean=predictorTest.mean()
predictorTestStd=predictorTest.std()
predictorTest_std= (predictorTest-predictorTestMean)/predictorTestStd #predictor test standardized


#####New model for the prostate cancer dataset by Principal Component Regression

#compute principal component decomposition
pca=PCA(n_components=8)
pca.fit(predictorTrain_std)
print('Dimension of Principal component matrix (V): ', pca.components_.shape) #right: V \in \Matrices(p,p)!

newTrain=np.dot(predictorTrain_std,pca.components_) #X * V
reg=LinearRegression(normalize=True).fit(newTrain,lpsaTrain) #OLS using the matrix of the transformed training set


#Show model coefficients, intercept and score
#I need to muliply the coefficients  by the matrix of the principal components, so I have the coefficients wrt the original variables
#See https://en.wikipedia.org/wiki/Principal_component_regression
origCoeff=pca.components_.dot(reg.coef_)    
print('PC regression coefficients are:', origCoeff)
print('PC regression intercept is:',reg.intercept_)
print('PC regression R^2 is:', reg.score(newTrain,lpsaTrain))

print('\n')


#Compute the PCR model with all possible numbers of principal components (from 1 to 8)
#and show model coeffs, intercepts and score
scoreTrain=[]
scoreTest=[]
for i in range (1,predictorTrain_std.shape[1]+1):
    pca=PCA(n_components=i)
    pca.fit(predictorTrain_std)
    newTrain=pca.transform(predictorTrain_std)
    newTest=pca.transform(predictorTest_std)
    reg=LinearRegression(normalize=True).fit(newTrain,lpsaTrain) #OLS using the matrix of the transformed training set
    #origCoeff=pca.components_.dot(reg.coef_)
    scoreTrain.append(reg.score(newTrain, lpsaTrain))
    scoreTest.append(reg.score(newTest,lpsaTest))

print('R square for different number of components: ', scoreTrain)
print('Intercep: ', reg.intercept_)


#Plot the scores on training set depending on the number of components used
plt.figure()
plt.plot(range(1,predictorTrain_std.shape[1]+1),scoreTrain,marker='*',label='Scores')
plt.xlabel('number of components')
plt.ylabel('R square')
plt.title('Score (R^2) versus number of components in the Train Set')
plt.legend()

plt.figure()
plt.plot(range(1,predictorTest_std.shape[1]+1),scoreTest,marker='*',label='Scores')
plt.xlabel('number of components')
plt.ylabel('R square')
plt.title('Score (R^2) versus number of components in the Test Set')
plt.legend()

    
#K-Fold cross validation on the entire dataset
N=10
kf=KFold(n_splits=N)
scoresTrainMeans = np.zeros(predictorTrain_std.shape[1])
errorTrainMeans = np.zeros(predictorTrain_std.shape[1])
errorTestMeans = np.zeros(predictorTrain_std.shape[1])

for train_i, test_i in kf.split(predictorTrain_std): #test_i will be used to k-Fold cross validation in the test set
    X_train, y_train = predictorTrain_std.iloc[train_i], lpsaTrain.iloc[train_i]
    X_test, y_test = predictorTrain_std.iloc[test_i],lpsaTrain.iloc[test_i] #split the train set in a train and a test set
    
    scoresTrain=[]
    errorTrain=[]
    errorTest=[]
    for i in range (1,predictorTrain_std.shape[1]+1):
        pca=PCA(n_components=i)
        pca.fit(X_train)
        newTrain=pca.transform(X_train) #Matrix of transformed training set
        newTest=pca.transform(X_test) #Matrix of transformed test
        
        
        set
        reg=LinearRegression().fit(newTrain,y_train) #OLS using the matrix of the transformed training set
        #origCoeff=pca.components_.dot(reg.coef_)
        scoresTrain.append(reg.score(newTrain, y_train))
        errorTrain.append(np.linalg.norm(y_train-reg.predict(newTrain)))
        errorTest.append(np.linalg.norm(y_test- reg.predict(newTest))) #y_test - y_predicted
        
    scoresTrainMeans+= scoresTrain 
    errorTrainMeans+=errorTrain
    errorTestMeans+=errorTest
    
scoresTrainMeans/=N
errorTrainMeans/=N #divide by the number of splits
errorTestMeans/=N

#plots
plt.figure()
plt.plot(range(1,predictorTrain_std.shape[1]+1),scoresTrainMeans,marker='*',label='scores')
plt.xlabel('number of components')
plt.ylabel('scores')
plt.title(str(N) + '-Folds Cross Validation Train Score with PCA')
plt.legend()

plt.figure()
plt.plot(range(1,predictorTrain_std.shape[1]+1),errorTrainMeans,marker='*',label='train error')
plt.title(str(N) + '-Folds Cross Validation Train Error with PCA')
plt.xlabel('number of components')
plt.ylabel('error on train set')
plt.legend()

plt.figure()
plt.plot(range(1,predictorTrain_std.shape[1]+1),errorTestMeans,marker='*',label='test error')
plt.xlabel('number of components')
plt.ylabel('scores')
plt.title(str(N) + '-Folds Cross Validation Test Error with PCA')
plt.legend()
plt.show()


#Clustering analysis


#df=pd.DataFrame(dataTrain,columns=['lweight','age'])
#kmeans=KMeans(n_clusters=4).fit(df)
#centroids=kmeans.cluster_centers_
#print(centroids)
#plt.figure()
#plt.scatter(df['lweight'], df['age'], c= kmeans.labels_.astype(float), alpha=0.5)
#plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
#plt.show()


WCSS= []

for i in range (1,9):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(dataTrain)
    WCSS.append(kmeans.inertia_)


plt.figure()
plt.plot(range(1, 9), WCSS,'-bx',label='distorsion(K)')
plt.axvline(3,0,max(WCSS),ls=':',label='best K')
plt.title('Elbow Method: distortion as a function of K')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.legend()
plt.show()