import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = yf.download(  # or pdr.get_data_yahoo(...
        # tickers list or string as well
        tickers = "BTC-USD ETH-USD DOGE-USD CNY=X DOW DX=F ^TNX GC=F ^IXIC SPY CL=F TSLA",

        # use "period" instead of start/end
        # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        # (optional, default is '1mo')
        period = "60d",
        interval="15m",

        # adjust all OHLC automatically
        # (optional, default is False)
        auto_adjust = True,

        # download pre/post regular market hours data
        # (optional, default is False)
        prepost = True,

        # use threads for mass downloading? (True/False/Integer)
        # (optional, default is True)
        threads = True,

        # proxy URL scheme use use when downloading?
        # (optional, default is None)
        proxy = None
    )

data['NextClose', 'BTC-USD'] = data['Close','BTC-USD'].shift(periods=-1)

data['prev_1', 'BTC-USD'] = data['Close','BTC-USD'].shift(periods=1)
data['prev_2', 'BTC-USD'] = data['Close','BTC-USD'].shift(periods=2)
data['prev_3', 'BTC-USD'] = data['Close','BTC-USD'].shift(periods=3)
data['prev_4', 'BTC-USD'] = data['Close','BTC-USD'].shift(periods=4)
data['prev_5', 'BTC-USD'] = data['Close','BTC-USD'].shift(periods=5)
data['prev_6', 'BTC-USD'] = data['Close','BTC-USD'].shift(periods=6)
data['prev_7', 'BTC-USD'] = data['Close','BTC-USD'].shift(periods=7)

data = data.dropna()

X = data.drop(['NextClose'], axis=1)
y = data['NextClose', 'BTC-USD']
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 14)
lr = LinearRegression().fit(X_train, y_train)
pred = lr.predict(X_test)

#print("lr.coef_: {}".format(lr.coef_))
#print("lr.intercept_: {}".format(lr.intercept_))

#print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
#print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

#for close, actual, prediction in zip(X_test['Close','BTC-USD'], y_test, pred):
#    print("previous close: " + str(close) +  " || actual: " + str(actual) + " || prediction: " + str(prediction))

current = yf.download(  # or pdr.get_data_yahoo(...
        # tickers list or string as well
        tickers = "BTC-USD ETH-USD DOGE-USD CNY=X DOW DX=F ^TNX GC=F ^IXIC SPY CL=F TSLA",

        # use "period" instead of start/end
        # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        # (optional, default is '1mo')
        period = "1d",
        interval="15m",

        # adjust all OHLC automatically
        # (optional, default is False)
        auto_adjust = True,

        # download pre/post regular market hours data
        # (optional, default is False)
        prepost = True,

        # use threads for mass downloading? (True/False/Integer)
        # (optional, default is True)
        threads = True,

        # proxy URL scheme use use when downloading?
        # (optional, default is None)
        proxy = None
    )

current['prev_1', 'BTC-USD'] = current['Close','BTC-USD'].shift(periods=1)
current['prev_2', 'BTC-USD'] = current['Close','BTC-USD'].shift(periods=2)
current['prev_3', 'BTC-USD'] = current['Close','BTC-USD'].shift(periods=3)
current['prev_4', 'BTC-USD'] = current['Close','BTC-USD'].shift(periods=4)
current['prev_5', 'BTC-USD'] = current['Close','BTC-USD'].shift(periods=5)
current['prev_6', 'BTC-USD'] = current['Close','BTC-USD'].shift(periods=6)
current['prev_7', 'BTC-USD'] = current['Close','BTC-USD'].shift(periods=7)



current = current.dropna()
current_pred = lr.predict(current)

print("Most Recent Price: " + str(current.iloc[-1][0]))
print("Projected Price in 15 Minutes: " + str(current_pred[-1]))

