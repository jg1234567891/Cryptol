# CryptOL
Using artificial intelligence to predict cryptocurrency price with Project CryptOL™ software application

Alexander Gribtsov
Department of Computer Science and Software Engineering
Monmouth University
400 Cedar Ave, West Long Branch, NJ 07764, United States of America	

Chris Woszczak
Department of Computer Science and Software Engineering
Monmouth University
400 Cedar Ave, West Long Branch, NJ 07764, United States of America	

John Guseman
Department of Computer Science and Software Engineering
Monmouth University
400 Cedar Ave, West Long Branch, NJ 07764, United States of America


Abstract
Background: Implementation of different types of artificial intelligence algorithms can produce fit data that predicts the price of different cryptocurrency types such as BitCoin.  
Related Work: Two related projects were observed for cross comparison of data performance and overall cryptocurrency price forecasting experience. 
Architecture: Project CryptOL is a web based full stack application. The design, implementation and deployment of this project is covered in detail.
Testing: Embedded into the CryptOL application as part of the design of the project is a classical false negative and false positive testing technique of fitness of running data. 
Discussion: During the development of the project all members of the team became familiar with most common data science terminology and functionality of standardized techniques. 
Conclusions: Project CryptOL is a great example of how standardized data science techniques and algorithms can be applied together to present a strong indicator of future cryptocurrency price.  

Keywords
Algorithms, artificial intelligence, bitcoin, cryptocurrency, machine learning, online machine learning, LSTM, neural networks, data science, survival modeling, time series, linear regression, logistic regression


# Approaches 
The goal of CryptOL is to use machine learning to predict the trend of Bitcoin prices. The project team selected three machine learning models for the experiment: linear regression, autoregressive integrated moving average (ARIMA), and long short-term memory (LSTM). To evaluate the success of the models, the project examines the directional accuracy of the predictions. That is, if a model predicts the price of Bitcoin will rise and it does rise, then a successful prediction has occurred. In addition, the focus of CryptOL is on short-term price predictions 15 minutes, 1 hour, and 1 day into the future. Cryptocurrency markets are highly volatile. Therefore, the project team concluded that it would be more difficult to arrive at long-term predictions.

While CryptOL uses three separate machine learning models, the project exhibits a singular design to tie the components together.

Data Generation
        CryptOL obtains its data using the Yahoo Finance API. The number of historical Bitcoin prices the project collects depends on the price prediction interval selected by the user. 15-minute predictions collect prices from the previous 60 days. This results in a data set with 5,760 price points ((1,440 minutes in a day X 60 days) / 15-minute intervals). 1-hour predictions collect prices from the previous 60 days resulting in a data set of 1,440 price points. Finally, the 1-day predictions collect prices from the previous 365 days. Unfortunately, this results in a data set with only 365 price points.
Splitting the Data
        Machine learning algorithms generally require that the data collected be split into at least two separate sets before a model is trained. One set is called the training set and the other is the test set. A machine learning model is trained using the training set. Predictions are made by applying the test set to the newly trained model.
          	CryptOL™ uses 80% of the data for training sets and 20% of the data for test sets.
Model Training, Predictions, & Evaluation
       This will be covered in the IMPLEMENTATIONS section of the paper.
User Interaction
        The user interacts with CryptOL through a graphical user interface. The GUI is built using a service called Anvil. Client requests are made on the front-end of the website. The python code used to make the predictions is run on the back-end of the Anvil service. The user selects an algorithm and a time interval if the option is offered. After some time, a prediction is returned to the user. At this point the user can decide whether to utilize the prediction result of the algorithm. 

Linear Regression
        CryptOL™ implemented a multiple linear regression model. Linear regression makes it possible to predict the value of an independent variable based on the values of one or more independent values, or features. For this project, the dependent variable is the price of Bitcoin in 15 minutes, 1 hour, or 1 day. The dependent variables are the features that the project team selected.
        This project employs the ordinary least squares method of linear regression. In the case of just two variables, one dependent and one independent, ordinary least squares regression seeks to fit a straight line through the plotted points. This fitted line is meant to model a relationship between the dependent and independent variables. The best fitting line is one that minimizes the distance between the values predicted by the line and the actual values of the regression model is by looking at the coefficient of determination, or R-Squared. The R-Squared ranges from 0 to 1. The number measures the amount of variance in the dependent variable caused by changes in the independent variable.
      While the CryptOL’s linear regression models often displayed R-Squared scores greater than 95%, this is likely not an ideal outcome. This is probably evidence that our models are overfitting the training dataset. In other words, the model exhibits desirable outcomes with the training data, however it reacts poorly when confronted with new data. 
     53 features were selected for the model. Three of these features are the prices of Bitcoin, Ether, and Dogecoin. The remaining 50 features include the 50 previous of Bitcoin at the selected time interval. The linear regression model trained using these features was able to predict the direction of the market with an accuracy as high as 72% sometimes.
Time Series (ARIMA)
CryptOL™ also implemented an Autoregressive Integrated Moving Average (ARIMA) model. The ARIMA model is used for time series forecasting. In general, a time series forecasting model will try and predict the future value of something based upon known past values.
          	There are three parts to the ARIMA model. Autoregressive refers to the number of lags used to determine the future value of something. For instance, is there a high correlation between the current value and the value one, two, three, etc. lags behind. For CryptOL, we saw the best results using a lag value of one. This means that if the user wishes to predict the price of bitcoin 15 minutes into the future, then CryptOL’s ARIMA model will pay close attention to the price in the immediate 15 minutes before.
          	The next part of ARIMA is the ‘integrated’ aspect. Generally, time series forecasting requires the data to be stationary. This means that the mean and variance of the sample data remain constant. Unfortunately, this is often not the case with Bitcoin prices. Integrating the values, in our case the price of Bitcoin, just means subtracting one value from its adjacent value. This differencing results in sample data that is stationary. This differencing can take place as many times as it is necessary to arrive at stationary data. The price of Bitcoin, like most time series data, only had to be differentiated once to achieve stationary data.
          	The final aspect of the ARIMA model is the moving average. It is possible to make the ARIMA take into account the moving average for a certain number of periods while making its prediction. CryptOL chose not to take into account the moving average for two reasons. First, experimenting with different moving averages did not drastically affect the results of the predictions. Second, including the moving averages greatly increased the time it took for the model to return a prediction.
     Long short-term memory (LSTM)
Long short-term memory (LSTM) is a form of recurrent neural networks (RNN). Unlike other RNNs, LSTM utilizes a cell state in order to store information over long periods of time. It does this by using a three gate system input gates, output gates, and forget gates, these gates maintain the flow of information over time. When compared with other RNNs, this cell state makes it advantageous when considering long-term predictions. Like ARIMA, this model is a time series forecasting model that can predict future values based on known previous values.

CryptOL implemented a sequential stacked LSTM model, using a one layered approach with 256 neurons. This model takes into consideration three features in order to make its prediction. They include the open, high, and close values at 15-minute intervals.

When creating a LSTM model, it is important to minimize the training loss & validation loss, while keeping the values of each close to prevent over or under fitting. To achieve this combination numerous factors were considered, tested, and optimized. However, hyperparameters such as number of epochs, number of neurons, and the type of activation function being used have shown in our case to correspond to the model's accuracy and consistency the most.

After testing, in our case, the optimal number of epochs to produce the best accuracy and consistency possible was 80. Higher numbers increased prediction time and showed no significant improvement in accuracy. Lower numbers decrease prediction time, but provide more inconsistent predictions.

As mentioned earlier, 256 neurons are used when compiling the model. There is not a one size fits all approach to determining the number of neurons, after experimenting with various other values such as 32, 64, 128, and 512. Smaller values decrease prediction time, but also decrease the consistency of  the model's accuracy. Values of 512 or higher made no significant improvements, and dramatically increased the time of prediction.

The only convincing options when considering activation function were Tanh and ReLU, as they both produced less loss than others while experimenting. ReLU showed a significant decrease in time taken to produce predictions. However, with this in mind, CryptOL™ decided to use the Tanh activation function because in our case, average accuracy is improved around 5% compared to ReLU. After all of the optimizations and many smaller tweaks, this model has predicted with accuracy as high as 68%.

Project Deployment
Once comfortable with our accuracy and consistency in predicting trends in the crypto market, CryptOL™ created its own web service via Anvil as previously mentioned. Anvil is able to provide our models with a reliable domain so investors and people with cryptocurrency interests alike can use them. Anvil was chosen because of its many features. The two most important include a built in database and the unique ability to implement all aspects of the project with strictly Python, making deployment simple when compared to other options.

When CryptOL™ is made available to the public, updates to our web service will stem directly from user feedback. These updates will occur biweekly inorder to please users and improve our product. Updates to improve or add models will be made quarterly to ensure that there is time for proper testing and experimentation.

# Research 
[1] 	Cohen, Gil. “Forecasting Bitcoin Trends Using Algorithmic Learning Systems.” Entropy 22, no. 8 (2020): 838. doi:10.3390/e22080838. 
[2] 	Lee, Brian K., Justin Lessler, and Elizabeth A. Stuart. “Improving Propensity Score Weighting Using Machine Learning.” Statistics in Medicine 29, no. 3 (2009): 337–46. doi:10.1002/sim.3782. 
[3] 	Munim, Ziaul Haque, Mohammad Hassan Shakil, and Ilan Alon. “Next-Day Bitcoin Price Forecast.” Journal of Risk and Financial Management 12, no. 2 (2019): 103. doi:10.3390/jrfm12020103. 
[4] 	“Introduction to Neural Networks for Time Series Forecasting.” Machine Learning for Time Series Forecasting with Python®, 2020, 137–65. doi:10.1002/9781119682394.ch5. 
[5] 	Chan, Phyllis, Xiaofei Zhou, Nina Wang, Qi Liu, René Bruno, and Jin Y. Jin. “Application of Machine Learning for Tumor Growth Inhibition – Overall Survival Modeling Platform.” CPT: Pharmacometrics & Systems Pharmacology 10, no. 1 (2020): 59–66. doi:10.1002/psp4.12576. 
[6] 	Tegen, Agnes. “Approaches to Interactive Online Machine Learning,” n.d. doi:10.24834/isbn.9789178770854. 
[7] 	Felizardo, Leonardo, Roberth Oliveira, Emilio Del-Moral-Hernandez, and Fabio Cozman. “Comparative Study of Bitcoin Price Prediction Using WaveNets, Recurrent Neural Networks and Other Machine Learning Methods.” 2019 6th International Conference on Behavioral, Economic and Socio-Cultural Computing (BESC), 2019. doi:10.1109/besc48373.2019.8963009. 
[8] 	Géron, Aurélien. Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems. Sebastopol,
         CA: O'Reilly Media, Inc., 2019. 
[9] 	Müller, Andreas Christian, and Sarah Guido. Introduction to Machine Learning with Python: a Guide for Data Scientists. Sebastopol: O'Reilly Media, 2018. 
