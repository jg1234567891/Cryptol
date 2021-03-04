#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd
from stockstats import *
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

get_ipython().run_line_magic('matplotlib', 'inline')


# In[46]:


pd.set_option('display.max_rows',10)
bc = StockDataFrame.retype(pd.read_csv('recent_btc.csv'))
pd.options.display.max_rows = 10
bc['open'] = bc['open (usd)']


# In[47]:


bc


# In[48]:


bc['rsi_7']


# In[49]:


bc['close_-1_r']


# In[50]:


bc['cr-ma1']


# In[51]:


bc['open_2_sma']


# In[52]:


bc['boll']


# In[53]:


new_bc = bc.drop(columns=['open (usd)', 'high','low','close_-1_s','close_-1_d','rs_7'])


# In[54]:


new_bc


# In[55]:


new_bc['next_close'] = new_bc['close'].shift(-1)


# In[56]:


dataset = new_bc.tail(2000)


# In[57]:


dataset = dataset.dropna()


# In[58]:


X = dataset[['close', 'rsi_7', 'close_-1_r','boll','open_2_sma', 'cr-ma1']]
y = dataset['next_close']


# In[59]:



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[60]:


regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[61]:


coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
coeff_df


# In[62]:


y_pred = regressor.predict(X_test)


# In[63]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})


# In[64]:


close_only = new_bc['close']


# In[65]:


df = df.join(close_only)


# In[66]:


df['downward_prediction'] =((df["Predicted"] < df["close"]))
df['correct_downward_prediction'] =((df["Predicted"] < df["close"]) & (df["Actual"] < df["close"]) )
df['upward_prediction'] =((df["Predicted"] > df["close"]))
df['correct_upward_prediction'] =((df["Predicted"] > df["close"]) & (df["Actual"] > df["close"]) )


# In[67]:


df.shape


# In[68]:


pd.options.display.max_rows = None
df


# In[69]:


downward = df[df['downward_prediction']] == True
correct_downward = df[df['correct_downward_prediction']] == True


# In[70]:


print("Downward Predictions: " + str(downward.shape[0]))
print("Correct Downward Predictions: " + str(correct_downward.shape[0]))


# In[71]:


upward = df[df['upward_prediction']] == True
correct_upward = df[df['correct_upward_prediction']] == True
print("Upward Predictions: " + str(upward.shape[0]))
print("Correct Upward Predictions: " + str(correct_upward.shape[0]))


# In[41]:


## RSI & CLOSING PRICE

for x in range(1,70):
    bc = StockDataFrame.retype(pd.read_csv('coinbase.csv'))
    rsi = 'rsi_' + str(x)
    rs = 'rs_' + str(x)
    bc[rsi]
    bc['close_-1_r']
    new_bc = bc.drop(columns=['open (usd)', 'high','low','close_-1_s','close_-1_d',rs])
    new_bc['next_close'] = new_bc['close'].shift(-1)
    dataset = new_bc.tail(600)
    dataset = dataset.dropna()
    
    X = dataset[rsi]
    
    y = dataset['next_close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
    coeff_df
    y_pred = regressor.predict(X_test)
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    close_only = new_bc['close']
    df = df.join(close_only)
    df['downward_prediction'] =((df["Predicted"] < df["close"]))
    df['correct_downward_prediction'] =((df["Predicted"] < df["close"]) & (df["Actual"] < df["close"]) )
    df['upward_prediction'] =((df["Predicted"] > df["close"]))
    df['correct_upward_prediction'] =((df["Predicted"] > df["close"]) & (df["Actual"] > df["close"]) )
    pd.options.display.max_rows = None
    df
    downward = df[df['downward_prediction']] == True
    correct_downward = df[df['correct_downward_prediction']] == True
    print("RSI: " + rsi + "\n")
    print("Downward Predictions: " + str(downward.shape[0]))
    print("Correct Downward Predictions: " + str(correct_downward.shape[0]))
    upward = df[df['upward_prediction']] == True
    correct_upward = df[df['correct_upward_prediction']] == True
    print("Upward Predictions: " + str(upward.shape[0]))
    print("Correct Upward Predictions: " + str(correct_upward.shape[0]) + '\n')
    print("Downward % Correct: " + str((correct_downward.shape[0]/downward.shape[0])*100))
    print("Upward % Correct: " + str((correct_upward.shape[0]/upward.shape[0])*100))
    print('\n=================================\n')


# In[ ]:


#RSI & ONE DAY % CHANGE & CLOSING PRICE

for x in range(1,40):
    bc = StockDataFrame.retype(pd.read_csv('coinbase.csv'))
    rsi = 'rsi_' + str(x)
    rs = 'rs_' + str(x)
    bc[rsi]
    bc['close_-1_r']
    new_bc = bc.drop(columns=['open (usd)', 'high','low','close_-1_s','close_-1_d',rs])
    new_bc['next_close'] = new_bc['close'].shift(-1)
    dataset = new_bc.tail(600)
    dataset = dataset.dropna()
    X = dataset[['close', rsi, 'close_-1_r']]
    y = dataset['next_close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
    coeff_df
    y_pred = regressor.predict(X_test)
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    close_only = new_bc['close']
    df = df.join(close_only)
    df['downward_prediction'] =((df["Predicted"] < df["close"]))
    df['correct_downward_prediction'] =((df["Predicted"] < df["close"]) & (df["Actual"] < df["close"]) )
    df['upward_prediction'] =((df["Predicted"] > df["close"]))
    df['correct_upward_prediction'] =((df["Predicted"] > df["close"]) & (df["Actual"] > df["close"]) )
    pd.options.display.max_rows = None
    df
    downward = df[df['downward_prediction']] == True
    correct_downward = df[df['correct_downward_prediction']] == True
    print("RSI: " + rsi + "\n")
    print("Downward Predictions: " + str(downward.shape[0]))
    print("Correct Downward Predictions: " + str(correct_downward.shape[0]))
    upward = df[df['upward_prediction']] == True
    correct_upward = df[df['correct_upward_prediction']] == True
    print("Upward Predictions: " + str(upward.shape[0]))
    print("Correct Upward Predictions: " + str(correct_upward.shape[0]) + '\n')
    print("Downward % Correct: " + str((correct_downward.shape[0]/downward.shape[0])*100))
    print("Upward % Correct: " + str((correct_upward.shape[0]/upward.shape[0])*100))
    print('\n=================================\n')


# In[ ]:




