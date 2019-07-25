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

#I want to use the method isoweekday() to compute the variables "day of the week" and "Hours". So I split the column "Date_1" in Date and Hours

split = data['Date_1'].str.split(" ", n = 1, expand = True) #split date and hour columns 
data['Date']=split[0] #assign the splitted variables to two new columns
data['Hours']=split[1]
data=data.drop('Date_1',axis=1) #drop old columns


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

#Now I add the columns to the dataframe. In order to do that, I have to create Series (which are columns in a DataFrame)
Weekday=pd.Series(weekdays).values
data['Weekday']=Weekday

Weekend=pd.Series(weekends).values
data['Weekends']=Weekend

Hour=pd.Series(hours).values
data['Hour']=Hour

#Add columns for year and month and day

Day=pd.Series(days).values
data['Day']=Day

Year=pd.Series(years).values
data['Year']=Year

Month=pd.Series(months).values
data['Month']=Month

data=data.drop('Date',axis=1)
data=data.drop('Hours',axis=1)
print(data.head())
print(data.tail())

#Plot of the correlation matrix
M=data.corr()
plt.matshow(M)
plt.show()  # better to add some labels 