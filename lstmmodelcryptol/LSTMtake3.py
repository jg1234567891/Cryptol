import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

tick = ["BTC-USD"]
def ts_download_btc(per="5d", inter="15m"):
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



df = ts_download_btc("5d")
df.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
df.dropna(inplace=True)
print(df)
cl = df
train = cl[0:int(len(cl)*0.80)]
scl = MinMaxScaler()

scl.fit(train.values.reshape(-1,1))
cl =scl.transform(cl.values.reshape(-1,1))
#Create a function to process the data into lb observations look back slices
# and create the train test dataset (90-10)
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
#Reshape data for (Sample,Timestep,Features) 
X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
#Fit model with history to check for overfitting
history = model.fit(X_train,y_train,epochs=30,validation_data=(X_test,y_test),shuffle=False)
model.summary() 


plt.figure(figsize=(12,8))
Xt = model.predict(X_train)
plt.plot(scl.inverse_transform(y_train.reshape(-1,1)), label="Actual")
plt.plot(scl.inverse_transform(Xt), label="Predicted")
plt.legend()
plt.title("Train Dataset")


plt.figure(figsize=(12,8))
Xt = model.predict(X_test)
plt.plot(scl.inverse_transform(y_test.reshape(-1,1)), label="Actual")
plt.plot(scl.inverse_transform(Xt), label="Predicted")
plt.legend()
plt.title("Test Dataset")

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
