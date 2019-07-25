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
from datetime import date 

data=pd.read_csv('dfSMDA.csv') #load data



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

Year=pd.Series(years).values
data['Year']=Year

Month=pd.Series(pd.to_numeric(months, errors='coerce')).values #I want 1 and not 01, and so on...
data['Month']=Month

data=data.drop('Date',axis=1)
data=data.drop('Hours',axis=1)
print(data.head())
print(data.tail())

#Plot of the correlation matrix
M=data.corr() 
plt.matshow(M)
plt.title('Correlation matrix')
#plt.show()  # better to add some labels

# **From the correlation matrix we can see that N_customers is highly correlated with: Beach_P_closed and Temperature **

plt.figure()
plt.scatter(data['Temp'],data['N_Customers'],marker='.',color='red') #IT'S A CLUSTERING
plt.xlabel('Temperature')
plt.ylabel('N_Customers')
plt.title('Number of customers depending on the temperature')
#plt.show()

# * * * * * * PLOT OF NUMBER OF CLIENTS (N_Customers variable) * * * * * *


#Scatter plot of the variable N_customers
plt.figure()
plt.scatter(range(1,data.shape[0]+1),data['N_Customers'],marker='.',s=2)
plt.title('First scatter plot of N_Customers')
#plt.show() #--> We can see that when the beach park is closed (1) we have lots of customers, while when the beach park is open (0) we have few customers

#Plot number of customers for every day of each month !!!


#Plot mean number of customers per month
nCustomersMean = np.zeros(12)
nCustomersStd=np.zeros(12)
M=range (1,13)
for m in M:
    CustomersPerMonth=data[data.Month==m]['N_Customers'] #take all the entries corresponding to month=m and then take the N_Customers mean (m=1--> all the januaries)
    nCustomersMean[m-1]=CustomersPerMonth.mean()
    nCustomersStd[m-1]=CustomersPerMonth.std()
    
plt.figure()
Month_label = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUNE', 'JULY', 'AUG', 'SEPT', 'OCT', 'NOV', 'DEC']
plt.bar(Month_label,nCustomersMean,color='white',edgecolor='blue')
plt.xlabel('Months')
plt.ylabel('Mean per Month')
plt.title('Mean of Number of customers per month')
#plt.show()

#Plot of the Mean number of customers per Hour in the Winter season
i = 1;
plt.figure(figsize=(30,10));
plt.suptitle('Mean number of customers per Hour in the Winter season')
H=range(0,24)
Day_label=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'] #for the plot
WinterSeasonMonths=[1,2,3,10,11,12] #jump from MAY to SEPTEMBER
for d in range(1,8):
    nCustomersPerHourMean=[]
    for h in H:
        CustomersPerHour=data[(data.Hour==h) & (data.Weekday==d) & (data.Month.isin(WinterSeasonMonths))]['N_Customers'] #number of clients at day d, hour h in the winter season
        nCustomersPerHourMean.append(CustomersPerHour.mean()) #contains the mean of customer for each hour for a given day and in winter season
        
    plt.subplot(4,2,i)
    plt.bar(H, height=nCustomersPerHourMean,color='white',edgecolor='blue')
    plt.title(Day_label[d-1])
    i+=1
plt.show()
# ** I can see that in weekends I have more customers, as expected **
# ** Not only, I can see that they are more in the evening **


# * * * * * * *PLOT TEMPERATURE (Temp variable) * * * * * * *




    
        
