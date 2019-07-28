import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import itertools
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from datetime import date
from scipy.stats import f



data=pd.read_csv('dfSMDA.csv') #load the dataset

# * * * * * * * CHANGE THE DATASET TO CREATE NEW VARIABLES * * * * * *

#I want to use the method isoweekday() to compute the variables "day of the week" and "Hours". So I split the column "Date_1" in Date and Hours

split = data['Date_1'].str.split(" ", n = 1, expand = True) #split date and hour columns. See https://www.geeksforgeeks.org/python-pandas-split-strings-into-two-list-columns-using-str-split/
data['Date']=split[0] #assign the splitted variables to two new columns
data['Hours']=split[1]
data=data.drop('Date_1',axis=1) #drop old Date columns


#Build the variable Weekday. The sintax is >> date(ANNO,MESE,GIORNO).isoweekday(), so I need yo cut off the "-" in each date by using split('-')
years= []
months= []
days= []

weekdays=[] #--> it will be the column Weekday
hours = [] #-->  it will be the column Hours
weekends=[] #--> it will be the column Weekends
for i in range (0,data.shape[0]):
    splitDate=data['Date'][i].split('-')
    splitHour=data['Hours'][i].split(':')
    years.append(splitDate[0])
    months.append(splitDate[1])
    days.append(splitDate[2])
    weekdays.append(date(int(years[-1]),int(months[-1]),int(days[-1])).isoweekday()) #every time I compute the date by looking at the last element in years
    if (weekdays[-1]== 6 or weekdays[-1]==7):
        weekends.append(1)
    else:
        weekends.append(0)

    hours.append(int(splitHour[0]))

#ADD THE NEW columns to the dataframe. In order to do that, I have to create Series (which are columns in a DataFrame)
Weekday=pd.Series(weekdays).values
data['Weekday']=Weekday

Weekend=pd.Series(weekends).values
data['Weekends']=Weekend

Hour=pd.Series(hours).values
data['Hour']=Hour

#ADD COLUMNS for day, year, month

Day=pd.Series(pd.to_numeric(days, errors='coerce')).values #I want 1 and not 01, and so on...
data['Day']=Day

Year=pd.Series(pd.to_numeric(years, errors='coerce')).values #I want 1 and not 01, and so on...
data['Year']=Year

Month=pd.Series(pd.to_numeric(months, errors='coerce')).values #I want 1 and not 01, and so on...
data['Month']=Month

data=data.drop('Date',axis=1)
data=data.drop('Hours',axis=1)
print(data.head())
print(data.tail())

#Plot of the correlation matrix
M=data.corr()
##plt.matshow(M)
##plt.title('Correlation matrix')
#####plt.show()  # better to add some labels

# **From the correlation matrix we can see that N_customers is highly correlated with: Beach_P_closed and Temperature **

########plt.figure()
##plt.scatter(data['Temp'],data['N_Customers'],marker='.',color='red') #IT'S A CLUSTERING
##plt.xlabel('Temperature')
##plt.ylabel('N_Customers')
##plt.title('Number of customers depending on the temperature')
######plt.show()




# * * * * * * PLOT OF NUMBER OF CLIENTS (N_Customers variable) * * * * * *


#Scatter plot of the variable N_customers
#######plt.figure()
##plt.scatter(range(1,data.shape[0]+1),data['N_Customers'],marker='.',s=2)
##plt.title('First scatter plot of N_Customers')
######plt.show() #--> We can see that when the beach park is closed (1) we have lots of customers, while when the beach park is open (0) we have few customers

#Plot number of customers for every day of each month !!


#Plot mean number of customers per month
nCustomersMean = np.zeros(12)
nCustomersStd=np.zeros(12)
M=range (1,13)
for m in M:
    CustomersPerMonth=data[data.Month==m]['N_Customers'] #take all the entries corresponding to month=m and then take the N_Customers mean (m=1--> all the januaries)
    nCustomersMean[m-1]=CustomersPerMonth.mean()
    nCustomersStd[m-1]=CustomersPerMonth.std()

#######plt.figure()
Month_label = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUNE', 'JULY', 'AUG', 'SEPT', 'OCT', 'NOV', 'DEC']
##plt.bar(Month_label,nCustomersMean,color='white',edgecolor='blue')
##plt.xlabel('Months')
##plt.ylabel('Mean per Month')
##plt.title('Mean of Number of customers per month')
######plt.show()

#Plot of the Mean number of customers per Hour in the Winter season
i = 1;
###plt.figure(figsize=(30,10));
###plt.suptitle('Mean number of customers per Hour in the Winter season')
H=range(0,24)
D=range(1,8)
Day_label=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'] #for the plot
WinterSeasonMonths=[1,2,3,10,11,12] #jump from MAY to SEPTEMBER
for d in D:
    nCustomersPerHourMean=[]
    for h in H:
        CustomersPerHour=data[(data.Hour==h) & (data.Weekday==d) & (data.Month.isin(WinterSeasonMonths))]['N_Customers'] #number of clients at day d, hour h in the winter season
        nCustomersPerHourMean.append(CustomersPerHour.mean()) #contains the mean of customer for each hour for a given day and in winter season

    ###plt.subplot(2,4,i)
    ###plt.bar(H, height=nCustomersPerHourMean,color='white',edgecolor='blue')
    ###plt.title(Day_label[d-1])
    i+=1
####plt.show()

# ** I can see that in weekends I have more customers, as expected **
# ** Not only, I can see that they are more in the evening **










# * * * * * * *PLOT TEMPERATURE (Temp variable) * * * * * * *

#Plot the temperature by day by hour
###plt.figure(figsize=(20,40))
###plt.suptitle('Mean temperature by Month and by Hours')
i=1
for m in M:
    Temperature= []
    for h in H:
        Temperatures=data[(data.Hour==h) & (data.Month==m)]['Temp']
        Temperature.append(Temperatures.mean())
    ##plt.subplot(3,4,i)
    ##plt.bar(H,Temperature,color='white',edgecolor='blue')
    ##plt.title(Month_label[m-1])
    i+=1
###plt.show()










# * * * * * * * * * PLOT BEACH PARK CLOSED (variable) * * * * * * *
#Beach_park_closed is a binary variable which is 1 if the Beach park is closed, 0 if the Beach park is open










# * * * * * *  * N_CUSTOMERS AND TEMPERATURE DISTIBUTIONS * * * * * * * *


###plt.figure(figsize=(30,10));
###plt.suptitle('Data Distributions')
###plt.subplot(2,1,1)
#sns.distplot(data['N_Customers'])
###plt.subplot(2,1,2)
#sns.distplot(data['Temp'])
#####plt.show()










# * * * * * * START REGRESSION * * * * * * * *

predictorsTrain=data[data.Year.isin([2016,2017])]
predictorsTest=data[data.Year==2018]       #--> Split the dataset in Train and Test Set

N_CustomersTrain=predictorsTrain['N_Customers'] #ytrain
predictorsTrain=predictorsTrain.drop('N_Customers',axis=1) #data train with only predictors
predictorsTrain=predictorsTrain.drop('Year',axis=1) #DROP YEAR BECAUSE THE MODEL MUST BE INDEPENDET FROM THE YEAR (otherwise I have NaN column)

N_CustomersTest=predictorsTest['N_Customers'] #ytest
predictorsTest=predictorsTest.drop('N_Customers',axis=1) #data test with only predictors
predictorsTest=predictorsTest.drop('Year',axis=1) #DROP YEAR BECAUSE THE MODEL MUST BE INDEPENDET FROM THE YEAR

# STANDARDIZE TRAIN AND TEST SET
predictorsTrain_std=(predictorsTrain - predictorsTrain.mean())/(predictorsTrain.std())
predictorsTest_std = (predictorsTest - predictorsTest.mean())/predictorsTest.std() #classical formula


#Regression with all the variables

print(' \n * * * * * * * * * * * REGRESSION * * * * * * * * * * *')
est=sm.OLS(N_CustomersTrain,sm.add_constant(predictorsTrain_std) ).fit()
print(est.summary()) #from this we can see that the Z-score(Month) < 2. In fact month is striclty correlated with temperature


print('\n Regression with ALL VARIABLES (except YEAR): ')
reg=LinearRegression().fit(predictorsTrain_std,N_CustomersTrain)
RSS=np.sum((N_CustomersTrain-reg.predict(predictorsTrain_std))**2)
print('\n Coefficients model: ', np.round(reg.coef_,3),'\n Intercept model: ', np.round(reg.intercept_,3))
print('\n Score: ', np.round(reg.score(predictorsTrain_std,N_CustomersTrain),4))

error=np.linalg.norm(N_CustomersTrain-reg.predict(predictorsTrain_std))

RMSETrain=np.sqrt(((N_CustomersTrain-reg.predict(predictorsTrain_std)) **2).mean())
RMSETest=np.sqrt(((N_CustomersTest-reg.predict(predictorsTest_std)) **2).mean())

print('\n RMSE on train: ', RMSETrain,'\n RMSE on test: ', RMSETest)

#Compute mean prediction error on Test data AND Base error rate on test data
ynewpred=reg.predict(predictorsTest)
meanPredErrTest=((ynewpred-N_CustomersTest).abs()).mean()
baseErrOnTest=((N_CustomersTrain.mean()-N_CustomersTest).abs()).mean()
print('\n Mean prediction error on test data: ', meanPredErrTest)
print('\n Base error rate on test data: ', baseErrOnTest)


#Now I consider as regressors: BEACH_PARK_CLOSED, TEMP, WEEKDAY, HOUR
print('\n - - - - - - - - - - - - \n')
print('\n Regression with: BEACH_PARK_CLOSED, TEMP, WEEKDAY, HOUR')

predictorsTrain1=predictorsTrain_std.drop(['Weekends','Day','Month'],axis=1)
predictorsTest1=predictorsTest_std.drop(['Weekends','Day','Month'],axis=1)

#print('\n Regression with: BEACH_PARK_CLOSED, TEMP, WEEKDAY, WEEKENDS, HOUR')
#predictorsTrain1=predictorsTrain_std.drop(['Month'],axis=1)
#predictorsTest1=predictorsTest_std.drop(['Month'],axis=1)

reg1=LinearRegression().fit(predictorsTrain1,N_CustomersTrain)
RSS1=np.sum((N_CustomersTrain-reg1.predict(predictorsTrain1))**2)

RMSE1_Train=np.sqrt(((N_CustomersTrain-reg1.predict(predictorsTrain1)) **2).mean())
RMSE1_Test=np.sqrt(((N_CustomersTest-reg1.predict(predictorsTest1)) **2).mean())
TestErrorOLS=np.linalg.norm(N_CustomersTest-reg1.predict(predictorsTest1))

print('\n Coefficients model: ', np.round(reg1.coef_,3),'\n Intercept model: ', np.round(reg1.intercept_,3)) #Magari printa coeff + str(num model) ,più carino !!
print('\n Score: ', np.round(reg1.score(predictorsTrain1,N_CustomersTrain),4))
print('\n RMSE on train: ', RMSE1_Train, '\n RMSE on test: ', RMSE1_Test)
print('\n OLS Regression Test Error: ', TestErrorOLS)

#Compute mean prediction error on Test data AND Base error rate on test data
ynewpred=reg1.predict(predictorsTest1)
meanPredErrTest=((ynewpred-N_CustomersTest).abs()).mean()
baseErrOnTest=((N_CustomersTrain.mean()-N_CustomersTest).abs()).mean()
print('\n Mean prediction error on test data: ', meanPredErrTest)
print('\n Base error rate on test data: ', baseErrOnTest)



#Using F-statistic in order to check that I can drop joint variables
#p1, p0=
#N=predictorsTrain_std.shape[0]
#F=((RSS1-RSS)/(3)) / ((RSS)/(N-6-1)) # F=((RSS_with_drop-RSS_no_drop)/(p1-p0))/(RSS_no_drop/(N-p1-p0))
#dfn, dfd=3 , predictorsTrain_std.shape[0]-6-1
#Cdf=f.cdf(F,dfn,dfd) #Pr(F_dfn,dfd> F), I use library scipy.stats
#p_value=1-Cdf #Definition of p-value
#print('\n p value for F statistic is :' , p_value)
# ** H_0: Model without DAY, MONTH is correct.
# ** Since p_value>0.05, then I do not reject the H_0 hyp


#Now I REMOVE BEACH_PARK_CLOSED
print('\n - - - - - - - - - - - - \n')
print('\n Regression without BEACH_PARK_CLOSED')
print('\n Regression with: TEMP, WEEKDAY, HOUR')


predictorsTrain2=predictorsTrain_std.drop(['Beach_Park_Closed','Weekends','Day','Month'],axis=1)
predictorsTest2=predictorsTest_std.drop(['Beach_Park_Closed','Weekends','Day','Month'],axis=1)

reg2=LinearRegression().fit(predictorsTrain2,N_CustomersTrain)
RSS2=np.sum((N_CustomersTrain-reg2.predict(predictorsTrain2))**2)

RMSE2_Train=np.sqrt(((N_CustomersTrain-reg2.predict(predictorsTrain2)) **2).mean())
RMSE2_Test=np.sqrt(((N_CustomersTest-reg2.predict(predictorsTest2)) **2).mean())

print('\n Coefficients model: ', np.round(reg2.coef_,3),'\n Intercept model: ', np.round(reg2.intercept_,3)) #Magari printa coeff + str(num model) ,più carino !!
print('\n Score: ', np.round(reg2.score(predictorsTrain2,N_CustomersTrain),4), '- - - - - -> ! LOW SCORE !')
print('\n RMSE on train: ', RMSE2_Train, '\n RMSE on test: ', RMSE2_Test)

#I can see that if I drop the binary variable BEACH_PARK_CLOSED I lose lot of information and in fact I have a low Train score (about 0.64).
#Say more please !!

# * * * * * *  * SUBSET SELECTION * * * * * * * *
#I choose:- Best subset selection
#         - Forward selection

print(' \n * * * * * * * * * * * SUBSET SELECTION * * * * * * * * * * *')
#With the OLS we can have large variance


#BEST SUBSET SELECTION
print('\n BEST SUBSET SELECTION \n' )
#typically we choose the smallest model that minimizes an estimate of the expected prediction error.

results=pd.DataFrame(columns=['num_features', 'features', 'MAE','RSS','R^2'])

for k in range (1,predictorsTrain_std.shape[1]+1):
    for subset in itertools.combinations(range(predictorsTrain_std.shape[1]),k):
        subset=list(subset)
        linreg=LinearRegression(normalize=True).fit(predictorsTrain_std.iloc[:,subset],N_CustomersTrain)
        linreg_pred=linreg.predict(predictorsTrain_std.iloc[:,subset]) #prediction using train set
        linreg_mae = np.mean(np.abs(N_CustomersTrain - linreg_pred))
        RSS=np.sum((N_CustomersTrain -linreg_pred)**2)
        Rsquare=linreg.score(predictorsTrain_std.iloc[:,subset] ,N_CustomersTrain)
        results = results.append(pd.DataFrame([{'num_features': k,
                                                'features': subset,
                                                'MAE': linreg_mae,
                                                'RSS':RSS,
                                                'R^2':Rsquare}]))#,sort=True)

#sort values by RSS
results_sort=results.sort_values('RSS') #riordina in base a RSS, che è ciò che voglio minimizzare

best_subset_model=LinearRegression(normalize=True).fit(predictorsTrain_std.iloc[:,results_sort['features'].iloc[0]],N_CustomersTrain) #fit(X_best,y_train)
best_subset_coefs=best_subset_model.coef_

print('Best subset Selection RSS : {}'.format(np.round(results_sort['RSS'].iloc[0],3)))
print('Best subset Selection R^2 : {}'.format(np.round(results_sort['R^2'].iloc[0],3)))


#Plot of subset size vs. RSS
results['min_RSS'] = results.groupby('num_features')['RSS'].transform(min) #aggiungo la colonna min_RSS
######plt.figure()
#ax=#plt.gca() #!!(Understand the method .gca() ) So I can have a logarithmic scale in y (RSS), because I have high magnitude
#ax.scatter(results.iloc[:,4],results.iloc[:,1],alpha=0.8,color='grey',s=5) #x:subset size, y:RSS
#ax.set_yscale('log')
##plt.ylabel('RSS for every model')
##plt.title('RSS - Best subset selection')
##plt.legend()
##plt.plot(results.iloc[:,4],results.iloc[:,-1],color='red',linestyle=':')

####plt.show()

#Plot of subset size vs. R^2
results['max_R^2']=results.groupby('num_features')['R^2'].transform(max) #aggiungo la colonna max R^2
#######plt.figure()
##plt.scatter(results.iloc[:,4],results.iloc[:,2],alpha=0.8,color='darkblue',s=5) #x:subset size, y:R^2
##plt.xlabel('k: subset size')
##plt.ylabel('R^2 for every model')
##plt.title('R^2 - Best subset selection')
##plt.legend()
##plt.plot(results.iloc[:,4],results.iloc[:,-1],color='green',linestyle=':')

####plt.show()

# ** From the last two plot we can see that we have the lowest RSS when we use a subset of k=4 variables. Moreover, looking at the corresponding
# "R^2 versus subset size" graph and this confirms the fact that we have the smallest R^2 when we have 4 subset. Moreover, this value corresponds
# to the value that we obtain when we drop the variables WEEKEND, DAY, MONTH **


print('\n - - - - - - - - - - - - \n')
print('\n FORWARD STEPWISE SELECTION \n' )

#FORWARD STEPWISE SELECTION:
#Forward Stepwise begins with a model containing no predictors, and then adds predictors to the model, one at the time.
#At each step, the variable that gives the greatest additional improvement to the fit is added to the model.

remaining_features=list(predictorsTrain_std.columns.values) #features to be included
features=[] #start with empty model
features_list=dict() #copy here features used
RSS_list=[]
RSquare_list=[]
for i in range (1,predictorsTrain_std.shape[1]+1):
    best_RSS=np.inf

    for combo in itertools.combinations(remaining_features,1):
        reg=LinearRegression().fit(predictorsTrain_std[list(combo) + features],N_CustomersTrain)
        pred=reg.predict(predictorsTrain_std[list(combo) + features]) #prediction to compute RSS
        RSS=np.sum((N_CustomersTrain -pred)**2)
        Rsquare=reg.score(predictorsTrain_std[list(combo) + features],N_CustomersTrain)

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
#######plt.figure()
##plt.scatter(np.arange(1,predictorsTrain_std.shape[1] + 1),RSS_list,label=' RSS ')
##plt.plot(np.arange(1,predictorsTrain_std.shape[1] + 1),RSS_list,'r-')
##plt.xlabel('k: subset size')
##plt.ylabel('RSS')
##plt.legend()
##plt.title('RSS - Forward stepwise selection')
####plt.show()






# * * * * * *  * SHRINKAGE METHODS * * * * * * * *
#I choose:- Ridge regressions
#         - Lasso


print(' \n * * * * * * * * * * * SHRINKAGE METHODS * * * * * * * * * * *')

print('\n RIDGE REGRESSION \n')



#reg=linear_model.Ridge(alpha=1.0)
#reg.fit(predictorsTrain_std,N_CustomersTrain)
#
#print('Ridge regression coefficients are:', reg.coef_)
#print('Ridge regression intercept is:', reg.intercept_)
#print('Ridge regression R^2 is:', reg.score(predictorsTrain_std,N_CustomersTrain))
#
#print('\n')

#Ridge coefficients as a function of the regularization parameters

alphas=np.logspace(-3,7,2000) #vector of regularization parameters. See documentation for logspace (goes from 1e-1 to 1e4 with 200 sample points)
coefs=list()
mycoefs=np.zeros([predictorsTest_std.shape[1],len(alphas)]) #predictorsTest_std.shape[1], the number of coefficients I expect
for i in range(len(alphas)):
    reg=linear_model.Ridge(alpha=alphas[i])
    reg.fit(predictorsTrain_std,N_CustomersTrain)
    #coefs.append(reg.coef_) #coef è un array di taglia 7. I do not use it because I can't display the legend in this way
    mycoefs[:,i]=reg.coef_ #matrix where for each column I have the 7 coefficients of the Ridge regression

#print(mycoefs.shape)
###plt.figure()
##plt.semilogx(alphas,mycoefs[5,:])   #just a check
###plt.show()

#PLOT THE COEFFICIENTS AGAINST alphas: cfr http://www.ds100.org/sp19/assets/lectures/regularization.pdf#Navigation24

##plt.figure()

#plt.semilogx(alphas,mycoefs[0,:],label='Beach P. Closed')
#plt.semilogx(alphas,mycoefs[1,:],label='Temperature')
#plt.semilogx(alphas,mycoefs[2,:],label='Weekday')
#plt.semilogx(alphas,mycoefs[3,:],label='Weekends')
#plt.semilogx(alphas,mycoefs[4,:],label='Hour')
#plt.semilogx(alphas,mycoefs[5,:],label='Day')
#plt.semilogx(alphas,mycoefs[6,:],label='Month')
#plt.xlabel('regularization parameter')
#plt.ylabel('coefficients')
#plt.legend() # old comment: "save coeff in a matrix to plot also the legend"
#plt.title('Regularization parameter against Ridge coeffs')
##plt.show()

# FIND the BEST REGULARIZATION PARAMETER by cross-validation
#predictorsTrain_std = predictorsTrain_std.drop(columns=['Weekday','Day','Month'])
#predictorsTest_std = predictorsTest_std.drop(columns=['Weekday','Day','Month'])  # I do it in the modified predictors sets

reg = linear_model.RidgeCV(alphas=np.logspace(-3,7,2000),store_cv_values=True)
reg.fit(predictorsTrain_std,N_CustomersTrain)
#print('Dimensions of cv_values: ', reg.cv_values_.shape)
print('\n best alpha: ', reg.alpha_)

#Compute the mean by column of cv_values_: I obtain the average cross-validation error for each alpha. This is the value that has to be minimized
#colMean=np.zeros([reg.cv_values_.shape[1],1])
#for i in range(reg.cv_values_.shape[1]):  # In order to avoid the loop I can use the method .mean() with argument axis
#    colMean[i]=reg.cv_values_[:,i].mean()

colMean=reg.cv_values_.mean(axis=0)

#Plot the leave-one-out cross validation error
#plt.subplot(2,1,1)
#plt.suptitle('Leave one out cross validation error and zooming')
#plt.loglog(alphas,colMean)  #high magnitude in alphas array
#plt.xlabel('reg. parameters')
#plt.ylabel('Average cross validation error')
#plt.subplot(2,1,2)
#plt.loglog(alphas[0:300],colMean[0:300]) #Zoom  where I have the minimum value of the curve. !! IN THE CASE I USE THE PREDICTORS WITH 4 VARIABLES
#plt.plot(alphas[200:500],colMean[200:500]) #Zoom IN THE CASE I USE THE WHOLE PREDICTOR TRAIN STANDARDIZED
#plt.xlabel('reg. parameters')
#plt.ylabel('Average cross validation error')

##plt.show()


#Identify the minimum leave-one-out cross-validation error and the related alpha, model coefficients and performance
minAv=np.min(colMean)
minIndex=np.where(colMean==minAv)[0][0] #tell me where colMean has its minimum
best_alpha=alphas[minIndex] #pass the index of the minimum

regRidge=linear_model.Ridge(alpha=best_alpha)
regRidge.fit(predictorsTrain_std,N_CustomersTrain)

RMSERidge_Train=np.sqrt(((N_CustomersTrain-regRidge.predict(predictorsTrain_std)) **2).mean())
RMSERidge_Test=np.sqrt(((N_CustomersTest-regRidge.predict(predictorsTest_std)) **2).mean())
TestErrorRidge=np.linalg.norm(N_CustomersTest-regRidge.predict(predictorsTest_std))

print('\n best_alpha by hand: ', best_alpha)

print('\n Ridge regression coefficients are:', regRidge.coef_)
print('\n Ridge regression intercept is:', regRidge.intercept_)
print('\n Ridge regression R^2 is:', regRidge.score(predictorsTrain_std,N_CustomersTrain))
print('\n RMSE on train: ', RMSERidge_Train, '\n RMSE on test: ', RMSERidge_Test)
print('\n Ridge Regression Test error: ', TestErrorRidge)
# ** I can see that the RMSE_Ridge_Test is LOWER THAN RMSE1_Test (60 vs 64) **

print('\n - - - - - - - - - - - - \n')


print('\n LASSO REGRESSION \n')
# * * * * * * START LASSO REGRESSION * * * * * * * *

#1. I use a cross validation in order to find the best alpha (regularization parameter). I call this regression regLasso.
#2. Do Lasso regression with the best regularization parameter I can find by using regLasso.alpha_
#   I call this model "reg"
#3. Compute RMSE on train and test set
#4. Plot Lasso coefficients as a function of the regularization parameter



#Find the best regularization parameter by cross-validation (see https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html#sphx-glr-auto-examples-linear-model-plot-lasso-model-selection-py)

regLasso = linear_model.LassoCV(cv=10).fit(predictorsTrain_std, N_CustomersTrain)
m_log_alphas = -np.log10(regLasso.alphas_)

##plt.figure()
#plt.plot(m_log_alphas, regLasso.mse_path_, ':') #mean square error
#plt.title('Mean square error for each fold')
#plt.plot(m_log_alphas, regLasso.mse_path_.mean(axis=1), 'k', #average across all folds
#         label='Average across the folds', linewidth=2)
#plt.axvline(-np.log10(regLasso.alpha_), linestyle='--', color='k',
 #           label='alpha: CV estimate') #show best regularization parameter in the plot

#plt.xlabel('-log(alpha)')
#plt.ylabel('Mean square error')
#plt.legend()
##plt.show()


#Now I do lasso regression with the best regularization parameter
#See https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_coordinate_descent_path.html#sphx-glr-auto-examples-linear-model-plot-lasso-coordinate-descent-path-py).
reg=linear_model.Lasso(alpha=regLasso.alpha_)
reg.fit(predictorsTrain_std,N_CustomersTrain)

RMSELasso_Train=np.sqrt(((N_CustomersTrain-reg.predict(predictorsTrain_std)) **2).mean())
RMSELasso_Test=np.sqrt(((N_CustomersTest-reg.predict(predictorsTest_std)) **2).mean())
TestErrorLasso=np.linalg.norm(N_CustomersTest-reg.predict(predictorsTest_std))



#Plot Lasso coefficients as a function of the regularization parameter !!
alphas_lasso, coefs_lasso, _=linear_model.lasso_path(predictorsTrain_std,N_CustomersTrain)
#plt.figure()

#for k in range(len(reg.coef_)):
    #plt.plot(alphas_lasso,coefs_lasso[k,:],linestyle='--')  !! decomment also the FOR loop in order to see the plot!
    #plt.legend()

#plt.xlabel('alpha')
#plt.ylabel('coefficients')
#plt.title('Lasso paths')
#plt.show()



#Show the best alpha, model coefficients and performance on training  set
print('\n')
print('\n The best alpha is: ',regLasso.alpha_)
print('\n Lasso regression intercept is: ',reg.intercept_)
print('\n Lasso regression coefficients are: ',reg.coef_)
print('\n Lasso regression R^2 is: ',reg.score(predictorsTrain_std,N_CustomersTrain))
print('\n RMSE on train: ', RMSELasso_Train, '\n RMSE on test: ', RMSELasso_Test)
print('\n Lasso Regression Test Error: ', TestErrorLasso)




# * * * * * *  * PRINCIPAL COMPONENT ANALYSIS* * * * * * * *

# I do it in the modified predictors sets
   #Remove less important columns !!
predictorsTrain_std = predictorsTrain_std.drop(['Weekday','Day','Month'],axis=1)
predictorsTest_std = predictorsTest_std.drop(['Weekday','Day','Month'],axis=1)   #Remove less important columns !!
#https://stackoverflow.com/questions/40389018/dropping-multiple-columns-from-a-data-frame-using-python/41579847

print(' \n * * * * * * * * * * * PRINCIPAL COMPONENT ANALYSIS * * * * * * * * * * *')


#1. Take all the components
#2. Loop on all the components and plot the scores on training set depending on the number of components used


print('\n - - - - - - - - - - - - \n')
print('\n Using all the components \n')


#compute principal component decomposition
pca=PCA(n_components=predictorsTrain_std.shape[1]) #4 components
pca.fit(predictorsTrain_std)
#print('Dimension of Principal component matrix (V): ', pca.components_.shape) #right: V \in \Matrices(p,p)!

newTrain=np.dot(predictorsTrain_std,pca.components_) #X * V
newTest=np.dot(predictorsTest_std,pca.components_)
reg=LinearRegression(normalize=True).fit(newTrain,N_CustomersTrain) #OLS using the matrix of the transformed training set


#Show model coefficients, intercept and score
#I need to muliply the coefficients  by the matrix of the principal components, so I have the coefficients wrt the original variables
#See https://en.wikipedia.org/wiki/Principal_component_regression
origCoeff=pca.components_.dot(reg.coef_)

print('\n PC regression coefficients are:', origCoeff)
print('\n PC regression intercept is:',reg.intercept_)
print('\n PC regression R^2 is:', reg.score(newTrain,N_CustomersTrain))


print('\n')



#Compute the PCR model with all possible numbers of principal components (from 1 to 8)
#and show model coeffs, intercepts and score
scoreTrain=[]
scoreTest=[]
for i in range (1,predictorsTrain_std.shape[1]+1):
    pca=PCA(n_components=i)
    pca.fit(predictorsTrain_std)
    newTrain=pca.transform(predictorsTrain_std)
    newTest=pca.transform(predictorsTest_std)
    reg=linear_model.LinearRegression().fit(newTrain,N_CustomersTrain) #OLS using the matrix of the transformed training set
    #origCoeff=pca.components_.dot(reg.coef_)
    scoreTrain.append(reg.score(newTrain, N_CustomersTrain))
    scoreTest.append(reg.score(newTest,N_CustomersTest))


print('R^2 for different number of components: ', scoreTrain)


#Plot the scores on training set depending on the number of components used
##plt.figure()
#plt.plot(range(1,predictorsTrain_std.shape[1]+1),scoreTrain,'b-',range(1,predictorsTrain_std.shape[1]+1),scoreTrain,'ro',label='Scores')
#plt.xlabel('number of components')
#plt.ylabel('R square')
#plt.title('Score (R^2) versus number of components in the Train Set')
#plt.legend()

##plt.figure()
#plt.plot(range(1,predictorsTest_std.shape[1]+1),scoreTest,'b-',range(1,predictorsTest_std.shape[1]+1),scoreTest,'ro',label='Scores')
#plt.xlabel('number of components')
#plt.ylabel('R square')
#plt.title('Score (R^2) versus number of components in the Test Set')
#plt.legend()

##plt.show()

# ** I can see that with 4 components I have the biggest R^2, as I expected from the previous analysis. For example, in subset selection I already saw that
#with 4 components I have a good R^2. Of course if I add other components it increases, but non in a significative way **

#K-Fold cross validation on the entire dataset, K=10 !!
N=10
kf=KFold(n_splits=N)
scoresTrainMeans = np.zeros(predictorsTrain_std.shape[1])
errorTrainMeans = np.zeros(predictorsTrain_std.shape[1])
errorTestMeans = np.zeros(predictorsTrain_std.shape[1])

for train_i, test_i in kf.split(predictorsTrain_std): #test_i will be used to k-Fold cross validation in the test set
    X_train, y_train = predictorsTrain_std.iloc[train_i], N_CustomersTrain.iloc[train_i]
    X_test, y_test = predictorsTrain_std.iloc[test_i], N_CustomersTrain.iloc[test_i] #split the train set in a train and a test set

    scoresTrain=[]
    errorTrain=[]
    errorTest=[]
    for i in range (1,predictorsTrain_std.shape[1]+1): #Now that I have splitted data, I do the PCA regression
        pca=PCA(n_components=i)
        pca.fit(X_train)
        newTrain=pca.transform(X_train) #Matrix of transformed training set
        newTest=pca.transform(X_test) #Matrix of transformed test

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
plt.plot(range(1,predictorsTrain_std.shape[1]+1),scoresTrainMeans,'b',range(1,predictorsTrain_std.shape[1]+1),scoresTrainMeans,'ro', label='scores')
plt.xlabel('number of components')
plt.ylabel('scores')
plt.title(str(N) + '-Folds Cross Validation Train Score with PCA')
plt.legend()

plt.figure()
plt.plot(range(1,predictorsTrain_std.shape[1]+1),errorTrainMeans,'b',range(1,predictorsTrain_std.shape[1]+1),errorTrainMeans,'ro', label='train error')
plt.title(str(N) + '-Folds Cross Validation Train Error with PCA')
plt.xlabel('number of components')
plt.ylabel('error on train set')
plt.legend()

plt.figure()
plt.plot(range(1,predictorsTrain_std.shape[1]+1),errorTestMeans,'b',range(1,predictorsTrain_std.shape[1]+1),errorTestMeans,'ro',label='test error')
plt.xlabel('number of components')
plt.ylabel('error on test set')
plt.title(str(N) + '-Folds Cross Validation Test Error with PCA')
plt.legend()
plt.show()


print(' \n * * * * * * * * * * * CLUSTERING ANALYSIS * * * * * * * * * * *')

WCSS= []

for i in range (1,data.shape[1]):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(data)
    WCSS.append(kmeans.inertia_)


plt.figure()
plt.plot(range(1, data.shape[1]), WCSS,'-bx',label='distorsion(K)')
plt.axvline(2,0,max(WCSS),ls=':',label='best K') #I saw "posthoc" by Elbow method that the good k is 2
plt.title('Elbow Method: distortion as a function of K')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.legend()
plt.show()

# ** I can see that the good k is k=2, by Elbow method **

km=KMeans(n_clusters=2)
km.fit(data) #I am interested in the km.labels_
columnsToCluster=[data['Temp'], data['N_Customers'], pd.DataFrame({ 'Cluster_Label': km.labels_})]
clusteredData = pd.concat(columnsToCluster, axis=1) # create a Dataframe with a column given by the labels

plt.figure()
plt.scatter(clusteredData['Temp'][clusteredData.Cluster_Label==1], clusteredData['N_Customers'][clusteredData.Cluster_Label==1],marker='.',s=0.8,label='First group') #First group
plt.scatter(clusteredData['Temp'][clusteredData.Cluster_Label==0], clusteredData['N_Customers'][clusteredData.Cluster_Label==0],marker='.',s=0.8,label='Second group') #Second group
plt.xlabel('Temp')
plt.ylabel('N_Customers')
plt.legend()
plt.show()
