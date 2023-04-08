from pmdarima import plot_acf, plot_pacf
import streamlit as st

from datetime import datetime
import numpy as np             #for numerical computations like log,exp,sqrt etc
import pandas as pd            #for reading & storing data, pre-processing
import matplotlib.pylab as plt #for visualization
           
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 6

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

def load_data():
    df=pd.read_excel('/Users/aameerkhan/Desktop/Streamlit-multipage/Nifty_monthly_OHLC.xlsx')
    df = df.dropna(True)
    df = df.drop_duplicates()
    
    return df

@st.cache
def preprocess_data(df):
    # Select only relevant columns
    df1=df.iloc[:,[0,1]].copy()
    
    # Convert 'Date' to Datetime and set it as index
    df1['Date'] = pd.to_datetime(df1['Date'])
    df1.set_index('Date', inplace=True)
    
    # Compute rolling statistics
    df1['rolmean'] = df1.rolling(window=12).mean()['Price']
    df1['rolstd'] = df1.rolling(window=12).std()['Price']

    # Compute log scale and moving averages
    df1_logScale = np.log(df1)
    df1['movingAverage'] = df1_logScale.rolling(window=12).mean()['Price']
    datasetLogScaleMinusMovingAverage = df1_logScale - df1['movingAverage']
    datasetLogScaleMinusMovingAverage.dropna(inplace=True)

    return df1, datasetLogScaleMinusMovingAverage
    
def plot_data(df1):
    # Plot original data and rolling statistics
    fig, ax = plt.subplots(figsize=(20,10))
    ax.plot(df1.index, df1['Price'], color='blue', label='Original')
    ax.plot(df1.index, df1['rolmean'], color='red', label='Rolling Mean')
    ax.plot(df1.index, df1['rolstd'], color='black', label='Rolling Std')
    ax.set_title('Rolling Mean & Standard Deviation')
    ax.set_xlabel('Date',fontsize=15)
    ax.set_ylabel('Price',fontsize=15)
    ax.legend(loc='best')
    st.pyplot(fig)

def plot_log_scale(df1):
    # Plot log scale and moving averages
    fig, ax = plt.subplots()
    ax.plot(df1.index, np.log(df1['Price']), label='Original')
    ax.plot(df1.index, df1['movingAverage'], color='#ff7f0e', label='Moving Average')
    ax.set_title('Log Scale & Moving Averages')
    ax.set_xlabel('Date',fontsize=15)
    ax.set_ylabel('Price (log scale)',fontsize=15)
    ax.legend(loc='best')
    st.pyplot(fig)

def plot_stationarity(timeseries):
    # Plot rolling statistics for stationarity test
    fig, ax = plt.subplots(figsize=(20,10))
    ax.plot(timeseries.index, timeseries, color='blue', label='Original')
    ax.plot(timeseries.index, timeseries.rolling(window=12).mean(), color='red', label='Rolling Mean')
    ax.plot(timeseries.index, timeseries.rolling(window=12).std(), color='black', label='Rolling Std')
    ax.set_title('Rolling Mean & Standard Deviation')
    ax.set_xlabel('Date',fontsize=15)
    ax.set_ylabel('Price (log scale)',fontsize=15)
    ax.legend(loc='best')
    st.pyplot(fig)

def test_stationarity(timeseries):
    # Compute rolling statistics and perform Dickey-Fuller test
    # Generate moving average and moving standard deviation
    movingAverage = timeseries.rolling(window=12).mean()
    movingSTD = timeseries.rolling(window=12).std()

    # Rename columns of movingAverage and movingSTD DataFrames to make them unique
    movingAverage.columns = [f'{col}_MA' for col in movingAverage.columns]
    movingSTD.columns = [f'{col}_MSD' for col in movingSTD.columns]

    # Concatenate the DataFrames and display the first 15 rows
    st.subheader('Rolling Statistics')
    st.write(pd.concat([timeseries, movingAverage, movingSTD], axis=1).head(15))


    dftest = adfuller(timeseries['Price'], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    
   

def main():
    st.set_page_config(page_title='Streamlit Demo: Time Series Analysis', layout='wide')
    st.title('Time Series Analysis Demo')
    
    # Load data
    df = load_data()
    
    # Preprocess data
    df1, datasetLogScaleMinusMovingAverage = preprocess_data(df)
    
    # Plot original data and rolling statistics
    plot_data(df1)
    
    # Plot log scale and moving averages
    plot_log_scale(df1)
    
    # Perform stationarity test and plot rolling statistics
    from statsmodels.tsa.stattools import adfuller

    def test_stationarity(timeseries):
        # perform Dickey-Fuller test and print the results
        dftest = adfuller(timeseries['Price'], maxlag=10)
        ...

    
    df1_logScale = np.log(df1)
    halflife = st.slider('Halflife', 0, 12, 6)
    min_periods = st.slider('Min Periods', 0, 10, 5)
    adjust = st.selectbox('Adjust', [True, False])
    exponentialDecayWeightedAverage = df1_logScale.ewm(halflife=halflife, min_periods=min_periods, adjust=adjust).mean()
    import matplotlib.pyplot as plt

  
  
    datasetLogScaleMinusExponentialMovingAverage = df1_logScale - exponentialDecayWeightedAverage
    datasetLogScaleMinusExponentialMovingAverage = datasetLogScaleMinusExponentialMovingAverage.dropna().drop_duplicates()
    test_stationarity(datasetLogScaleMinusExponentialMovingAverage)

    datasetLogDiffShifting = df1_logScale - df1_logScale.shift()
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(datasetLogDiffShifting, label='Time Shift Transformation')
    ax.legend(loc='best')
    st.pyplot(fig)
    df1_logScale = df1_logScale.dropna().drop_duplicates()
    datasetLogDiffShifting = datasetLogDiffShifting.dropna().drop_duplicates()
    test_stationarity(datasetLogDiffShifting)
    seasonal = 12
    decomposition = seasonal_decompose(df1_logScale.index,period=seasonal) 
    residual = decomposition.resid
    
    
   

    
    decomposedLogData = residual
    test_stationarity(decomposedLogData)
    
    lag_acf = acf(datasetLogDiffShifting, nlags=20)
    lag_pacf = pacf(datasetLogDiffShifting, nlags=20, method='ols')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(datasetLogDiffShifting, ax=ax1, lags=20)
    plot_pacf(datasetLogDiffShifting, ax=ax2, lags=20)
    st.pyplot(fig)
    
    model_type = st.selectbox('Model Type', ['ARIMA', 'ARMA', 'MA'])
    if model_type == 'ARIMA':
        p = st.slider('P', 0, 5, 2)
        d = st.slider('D', 0, 2, 1)
        q = st.slider('Q', 0, 5, 1)
        model = ARIMA(df1_logScale, order=(p,d,q))
    
    elif model_type == 'ARMA':
        p = st.slider('P', 0, 5, 2)
        d = st.slider('D', 0, 2, 1)
        q = st.slider('Q', 0, 5, 1)
        model = ARIMA(df1_logScale, order=(p,d,q))

    else:
        p = st.slider('P', 0, 5, 1)
        d = st.slider('D', 0, 2, 1)
        q = st.slider('Q', 0, 5, 2)
        model = ARIMA(df1_logScale, order=(p,d,q))

    results_ARIMA = model.fit(disp=-1)
    
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(datasetLogDiffShifting, label='Dataset')
    ax.plot(results_ARIMA.fittedvalues, color='red', label='Fitted Values')
    ax.set_title('RSS: %.4f'%sum((results_ARIMA.fittedvalues - datasetLogDiffShifting['Price'])**2))
    ax.legend(loc='best')
    st.pyplot(fig)
    
    predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    
    predictions_ARIMA_log = pd.Series(df1_logScale['Price'].iloc[1], index=df1_logScale.index)
    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)

    # Inverse of log is exp.
    predictions_ARIMA = np.exp(predictions_ARIMA_log)
    
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(df['Date'], df['Price'], label='Actual Price')
    ax.plot(df['Date'], predictions_ARIMA, color='red', label='Predicted Price')
    ax.legend(loc='best')
    st.pyplot(fig)
    
    df1_logScale = np.log(df1)
    model = ARIMA(df1_logScale, order=(2,1,0))
    results_AR = model.fit(disp=-1)

    plt.figure(figsize=(10,5))
    plt.plot(df1_logScale)
    plt.plot(results_AR.fittedvalues, color='red')
    plt.title('AR Model: RSS %.4f'% sum((results_AR.fittedvalues - df1_logScale)**2))

    st.pyplot()
    df1_logScale = np.log(df1)

    predictions_ARIMA_diff = pd.Series(results_AR.fittedvalues, copy=True)
    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    predictions_ARIMA_log = pd.Series(df1_logScale.iloc[0], index=df1_logScale.index)
    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
    predictions_ARIMA = np.exp(predictions_ARIMA_log)

    plt.figure(figsize=(10,5))
    plt.plot(df1)
    plt.plot(predictions_ARIMA)
    plt.title('Actual vs. Predicted Price')

    st.pyplot()

    results_ARIMA = model.fit(disp=-1)
    x = results_ARIMA.plot_predict(1, 200)

    st.pyplot()

if __name__ == '__main__':
    main()

