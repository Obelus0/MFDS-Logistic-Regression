#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import math
import sklearn
import matplotlib.pyplot as plt

'''
read the excel file into code
'''
df=pd.read_excel('Dataset_Question2.xlsx')
'''
create dummy variables for categorical 'Test' column

'''
dummy=pd.get_dummies(df['Test'])

'''
create data frames

'''

df1=pd.concat((df,dummy),axis=1)
df2=df1.drop(['Test'],axis=1)
df3=df2.drop(['Fail'],axis=1)

'''
keeping only one column of 'Test'~'Pass or fail' and then renaming it 'Test'
levels~PASS=1,FAIL=0

'''
df3=df3.rename(columns={"Pass":"Test"})
x=df3[['Temperature','Pressure','Feed Flow rate','Coolant Flow rate','Inlet reactant concentration']]
y=df3[['Test']]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

X=x_train.iloc[:,].values
Y=y_train.iloc[:,0].values
Xt=x_test.iloc[:,].values

'''
assuming normal distribution making data centred about mean for training data

'''
import numpy
X_std=numpy.copy(X)
X_std[:,0]=(X_std[:,0]-X_std[:,0].mean())/X_std[:,0].std()
X_std[:,1]=(X_std[:,1]-X_std[:,1].mean())/X_std[:,1].std()
X_std[:,2]=(X_std[:,2]-X_std[:,2].mean())/X_std[:,2].std()
X_std[:,3]=(X_std[:,3]-X_std[:,3].mean())/X_std[:,3].std()
X_std[:,4]=(X_std[:,4]-X_std[:,4].mean())/X_std[:,4].std()

'''
assuming normal distribution making data centred about mean for test data

'''
import numpy
X_t_std=numpy.copy(Xt)
X_t_std[:,0]=(X_t_std[:,0]-X_t_std[:,0].mean())/X_t_std[:,0].std()
X_t_std[:,1]=(X_t_std[:,1]-X_t_std[:,1].mean())/X_t_std[:,1].std()
X_t_std[:,2]=(X_t_std[:,2]-X_t_std[:,2].mean())/X_t_std[:,2].std()
X_t_std[:,3]=(X_t_std[:,3]-X_t_std[:,3].mean())/X_t_std[:,3].std()
X_t_std[:,4]=(X_t_std[:,4]-X_t_std[:,4].mean())/X_t_std[:,4].std()


'''
Creating a sigmoid function that returns values between on '0' and '1'
'''
def sigmoid(X,theta):
    z=np.dot(X,theta[1:])+theta[0]
    return 1.0/(1.0+np.exp(-z))

'''
create an appropriate cost function that maximizes the probability for data or minimizes 
negative of that cost function

'''

def costfunction(hx,Y):
    L=-Y.dot(numpy.log(hx))-((1-Y).dot(numpy.log(1-hx)))
    return L

'''
CREATE AN EMPTY LIST NAMED COST THAT STORES VALUES FOR COST FUNCTION OVER THE NUMBER OF ITERATIONS
'''

'''
Creating a logistic regression gradient descent function that takes initial guesses of weights and uses
gradient descent to obtain optimum weights

'''


def lrgradient(X,y,theta,alpha,num_iter):
    '''
    CREATE AN EMPTY LIST NAMED COST THAT STORES VALUES FOR COST FUNCTION OVER THE NUMBER OF ITERATIONS
    '''
    cost=[]
    for i in range(num_iter):
        hx=sigmoid(X,theta)
        error=hx-Y
        grad=np.dot(X.T,error)
        p=error.sum()
        theta[1:]=theta[1:]-alpha*grad
        theta[0]=theta[0]-alpha*p
        cost.append(costfunction(hx,Y))
    return cost
'''
giving learning rate a value

'''
alpha=0.00175
'''
giving number of iterations

'''
num_iter=300

m,n=X.shape
'''
initiallization to all weights equal to zero

'''
theta=np.zeros(n+1)
'''
adding lrgradient output to empty list called cost

'''
cost=lrgradient(X_std,y,theta,alpha,num_iter)
'''
creating an array of predicted Y based on training set 
'''
hx=sigmoid(X_t_std,theta)

'''
giving a threshold to convert predicted Y into binaries

'''
for i in range(0,len(x_test)):
    if hx[i]>=0.5:
        hx[i]=1
    elif hx[i]<0.5:
        hx[i]=0
hx=numpy.array(hx,dtype='int')
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
"""
Creating a confusion matrix of test outcomes for accuracy and comparison

"""
results=confusion_matrix(y_test,hx)
print("confusion matrix:")
print(results)
fig=plt.figure()
ax=fig.add_subplot()
cax=ax.matshow(results)
fig.colorbar(cax)
plt.title('CONFUSION MATRIX')
plt.xlabel('PREDICTED VALUES')
plt.ylabel('TRUE VALUES')
plt.show()
'''
showing accuracy of test model
'''
print('accuracy score:',accuracy_score(y_test,hx)*100)
'''
printing precision,recall,f1 score for test data.

'''
print('report')
print(classification_report(y_test,hx))
import seaborn as sn
'''
plotting cost function as function of number of iterations

'''
plt.plot(range(1,len(cost)+1),cost)
plt.xlabel("NO OF ITERATIONS")
plt.ylabel("COST")
plt.title("LOGISTIC REGRESSION-for learning rate alpha="+" "+str(alpha)+" "+"and initialization of 0")
plt.show()
print('THE EQUATION IS:')
print(str(theta[0])+str(theta[1])+'*'+'X1'+str(theta[2])+'*'+'X2'+str(theta[3])+'*X3'+'+'+str(theta[4])+'*X4'+str(theta[5])+'*X5')


# In[22]:


'''
Calculating logistic regression using inbuilt statsmodel package to crossvalidate and
calculate significance level of each parameter
'''
import statsmodels.api as sm
xvar=sm.add_constant(X_std)
regression=sm.Logit(y_train,xvar).fit()
print(regression.summary())


# In[10]:


'''
code for scatter plots between independant variables
'''
for i in range(1,5):
    plt.scatter(X_t_std[:,0],X_t_std[:,i])
    title1=['Temperature','Pressure','Feed Flow rate','Coolant Flow rate','Inlet reactant concentration']
    plt.xlabel(title1[0])
    plt.ylabel(title1[i])
    plt.show()
for i in range(2,5):
    plt.scatter(X_t_std[:,1],X_t_std[:,i])
    title1=['Temperature','Pressure','Feed Flow rate','Coolant Flow rate','Inlet reactant concentration']
    plt.xlabel(title1[1])
    plt.ylabel(title1[i])
    plt.show()
for i in range(3,5):
    plt.scatter(X_t_std[:,2],X_t_std[:,i])
    title1=['Temperature','Pressure','Feed Flow rate','Coolant Flow rate','Inlet reactant concentration']
    plt.xlabel(title1[2])
    plt.ylabel(title1[i])
    plt.show()
for i in range(4,5):
    plt.scatter(X_t_std[:,3],X_t_std[:,i])
    title1=['Temperature','Pressure','Feed Flow rate','Coolant Flow rate','Inlet reactant concentration']
    plt.xlabel(title1[3])
    plt.ylabel(title1[i])
    plt.show()


# In[13]:


'''
get the qq plot for the for each independant variable

'''
import statsmodels.api as sm
import pylab as py
for i in range(0,5):
    sm.qqplot(X_std[:,4],line='45')
    title1=['Temperature','Pressure','Feed Flow rate','Coolant Flow rate','Inlet reactant concentration']
    py.title('q-q plot for '+str(title1[i]))
    py.show()


# In[ ]:




