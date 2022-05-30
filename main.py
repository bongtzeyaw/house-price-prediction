import numpy as np 
import pandas as pd

import plotly.express as px 
import plotly.graph_objects as go
import plotly.io as pio
pio.templates

import seaborn as sns 
import matplotlib.pyplot as plt 

from scipy import stats
from scipy.stats import norm, skew #for some statistics

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston 

if __name__ == "__main__":
    ### Load and preprocess data

    load_boston = load_boston() 
    X = load_boston.data 
    y = load_boston.target 
    data = pd.DataFrame(X, columns=load_boston.feature_names) 
    data["SalePrice"] = y # saleprice 
    data.head() 
    compression_opts = dict(method='zip',
                            archive_name='out.csv')  
    data.to_csv('out.zip', index=False,
            compression=compression_opts)  
    #print(load_boston.DESCR) # Uncomment to see data set characteristics
    #data.describe() #Uncomment to see basic statistical analysis of data

    ### Exploratory Data Analysis (EDA)

    # Carry out pairplot
    data.isnull().sum()
    sns.pairplot(data, height=2.5)
    plt.tight_layout()

    # Visualise skewness and kurtosis of data
    sns.distplot(data['SalePrice'])
    print("Skewness: %f" % data['SalePrice'].skew())
    print("Kurtosis: %f" % data['SalePrice'].kurt())

    # visualise scatter plot
    fig, ax = plt.subplots()
    ax.scatter(x = data['CRIM'], y = data['SalePrice'])
    plt.ylabel('SalePrice', fontsize=13)
    plt.xlabel('CRIM', fontsize=13)
    plt.show()
    fig, ax = plt.subplots()
    ax.scatter(x = data['AGE'], y = data['SalePrice'])
    plt.ylabel('SalePrice', fontsize=13)
    plt.xlabel('CRIM', fontsize=13)
    plt.show()

    # Visualise sale price distribution and probability plot
    sns.distplot(data['SalePrice'] , fit=norm);
    (mu, sigma) = norm.fit(data['SalePrice'])
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                loc='best')
    plt.ylabel('Frequency')
    plt.title('SalePrice distribution')
    fig = plt.figure() #Get also the QQ-plot
    res = stats.probplot(data['SalePrice'], plot=plt)
    plt.show()
 
    data["SalePrice"] = np.log1p(data["SalePrice"])
    sns.distplot(data['SalePrice'] , fit=norm);
    (mu, sigma) = norm.fit(data['SalePrice'])
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                loc='best')
    plt.ylabel('Frequency')
    plt.title('SalePrice distribution')
    fig = plt.figure()
    res = stats.probplot(data['SalePrice'], plot=plt)
    plt.show()

    ### Data correlation

    plt.figure(figsize=(10,10))
    cor = data.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.PuBu)
    plt.show()

    # Take absolute value of the correlation 
    cor_target = abs(cor["SalePrice"]) 

    # Filter highly correlated features
    relevant_features = cor_target[cor_target>0.2]  

    # Get the names of the features 
    names = [index for index, value in relevant_features.iteritems()]

    # Remove target feature
    names.remove('SalePrice')  

    # Display the features 
    print(names) 
    print(len(names))

    ### Build model

    # Testing set is 20% of raw data
    X = data.drop("SalePrice", axis=1) 
    y = data["SalePrice"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    # Fit the model and predict
    lr = LinearRegression() 
    lr.fit(X_train, y_train)
    predictions = lr.predict(X_test)  
    print("Actual value of the house:- ", y_test[0]) 
    print("Model Predicted Value:- ", predictions[0])

    # Evaluate the error. May also use other errors such as mae
    mse = mean_squared_error(y_test, predictions) 
    rmse = np.sqrt(mse)
    print("Root mean square error:- ", rmse)