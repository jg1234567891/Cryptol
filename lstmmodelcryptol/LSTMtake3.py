import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


tick = ["BTC-USD"]
def ts_download_btc(per="3d", inter="15m"):
    data = yf.download(
            tickers = tick, 
            period = per,
            interval = inter,
            auto_adjust = True,
            prepost = True,
            threads = True,
            proxy = None
            )
    return data[["Open","High","Close"]]



df = ts_download_btc("3d")
df.fillna(method="ffill", inplace=True)
df.dropna(inplace=True)
print(df)
cl = df
train = cl[0:int(len(cl)*0.80)]
scl = MinMaxScaler()
scl.fit(train.values.reshape(-1,1))
cl =scl.transform(cl.values.reshape(-1,1))
def processData(data,lb):
    X,Y = [],[]
    for i in range(len(data)-lb-1):
        X.append(data[i:(i+lb),0])
        Y.append(data[(i+lb),0])
    return np.array(X),np.array(Y)

lb=10
X,y = processData(cl,lb)
X_train,X_test = X[:int(X.shape[0]*0.90)],X[int(X.shape[0]*0.90):]
y_train,y_test = y[:int(y.shape[0]*0.90)],y[int(y.shape[0]*0.90):]

model = Sequential()
model.add(LSTM(256,input_shape=(lb,1)))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')
X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
history = model.fit(X_train,y_train,epochs=80,validation_data=(X_test,y_test),shuffle=False)
model.summary() 

Xt = model.predict(X_train)
#train data
Xt = model.predict(X_test)
#test data

pred = scl.inverse_transform(Xt).tolist()
acc = 0
for x in range(len(Xt)-1):
    if(pred[x] > Xt[x] and Xt[x+1]>Xt[x]):
        acc += 1
    elif (pred[x] < Xt[x] and Xt[x+1]<Xt[x]):
        acc += 1


accu = (acc/len(Xt)*100)

print("prediction: ")
print(pred[-1])
print("accu = ", accu)
