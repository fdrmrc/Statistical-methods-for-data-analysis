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

print(data.head())

#Build the variable Weekday. The sintax is >> date(2019,7,25).isoweekday(), so I need yo cut off the "-" in each date
