from http.client import REQUEST_ENTITY_TOO_LARGE
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, jsonify, request
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels as sm
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA



st.set_option('deprecation.showPyplotGlobalUse', False)
def hash_complex(x):
    return hash((x.real, x.imag))

@st.cache(show_spinner=True,allow_output_mutation=True, hash_funcs={complex: hash_complex})
def fit_arima_model(data):
    return ARIMA(data, order=(1, 1, 1)).fit()
# Define the Flask app
app = Flask(__name__)
st.header("Nifty 50 Data Analysis and Prediction")
# Define the paths and directories
ROOT_DIR = "/kaggle/input"
DATA_DIR = "/Users/aameerkhan/Desktop/Streamlit-multipage/archive"

# Get list of CSV filenames from input directory
csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

# Create a dropdown menu to select a file
selected_file = st.sidebar.selectbox("Select a CSV file", csv_files)

if selected_file:
    # Load the selected CSV file into a Pandas dataframe
    df = pd.read_csv(os.path.join(DATA_DIR, selected_file))
    df = df.dropna().drop_duplicates()
    
    st.text("")
        


   

    # Extract the close price ser¯ies and perform stationarity test
    df_close = df['Close']
    st.subheader(selected_file)
    st.subheader('Data')
    st.dataframe(df)
    st.text("")
    
    @st.cache   # this decorator will cache the function result for better performance of app
    def RSI(df,n=14):
        close = df['Close']
        delta = close.diff()
        delta = delta[1:]
        pricesUp = delta.copy()
        pricesDown = delta.copy()
        pricesUp[pricesUp<0]=0
        pricesDown[pricesDown>0]=0
        rollUp = pricesUp.rolling(n).mean()
        rollDown = pricesDown.abs().rolling(n).mean()
        rs = rollUp/rollDown
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi



    # fill missing values with zero
    df.fillna(0, inplace=True)
            # Display OHLC plot
    st.subheader("OHLC Plot")
    fig_ohlc = make_subplots(rows=1, cols=1)
    fig_ohlc.add_trace(go.Ohlc(x=df.Date, open=df.Open, high=df.High, low=df.Low, close=df.Close, name='Price'))
    st.plotly_chart(fig_ohlc)

    # Display volume plot
    st.subheader("Volume Plot")
    fig_volume = make_subplots(rows=1, cols=1)
    fig_volume.add_trace(go.Scatter(x=df.Date, y=df.Volume, name='Volume'))
    st.plotly_chart(fig_volume)
    # call the RSI function and store it in 'RSI' column of dataframe
    df['RSI'] = RSI(df).fillna(0)

    # create figure
    fig = go.Figure(go.Scatter(x=df['Date'],y=df['RSI'],name='RSI'))

    st.subheader("Relative Strength Index Plot")

    # display the figure using Streamlit
    st.plotly_chart(fig)
    def test_stationarity(timeseries):
        # Calculate rolling statistics
        rolmean = timeseries.rolling(12).mean()
        rolstd = timeseries.rolling(12).std()

        # Plot the rolling statistics
        st.subheader("Rolling mean and standard deviation")
        fig, ax = plt.subplots()
        ax.plot(timeseries, color='blue', label='Original')
        ax.plot(rolmean, color='red', label='Rolling Mean')
        ax.plot(rolstd, color='black', label='Rolling Std')
        ax.legend(loc='best')
        ax.set_label('Rolling Mean and Standard Deviation')
        st.pyplot(fig)
        st.text("")

        # Perform augmented Dickey-Fuller test
        adft = adfuller(timeseries, autolag='AIC')

        output = pd.Series(adft[0:4], index=['Test Statistics', 'p-value',
                                              'No. of lags used',
                                              'Number of observations used'])
        for key, values in adft[4].items():
            output[f'critical value ({key})'] = values
        st.subheader('ADF-Test Statistics')
        st.write(output)
        st.text("")

    test_stationarity(df_close)

    # Decompose the time series data into trend, seasonality, and residual components
    result = seasonal_decompose(df_close, model='multiplicative', period=13)

    # Train an auto ARIMA model on the training data
    train_data, test_data = df_close[3:int(len(df_close)*0.9)], df_close[int(len(df_close)*0.9):]
    model = auto_arima(train_data, start_p=0, start_q=0,
                        test='adf',
                        max_p=3, max_q=3,
                        m=1,
                        d=None,
                        seasonal=False,
                        start_P=0,
                        D=0,
                        trace=True,
                        error_action='ignore',
                        suppress_warnings=True,
                        stepwise=True)

    # Define the endpoints
    @app.route('/autoarima', methods=['POST'])
    def autoarima():
        # Get data from the request payload
        data = request.json['data']

        # Make predictions
        forecast = model.predict(n_periods=len(data))

        # Return the forecasted values as JSON
        return jsonify({'forecast': forecast.tolist()})

    st.subheader('Actual Model Summary')
    st.text(str(model.summary()))
    from sklearn.metrics import mean_squared_error
    import math

    # Make predictions
    forecast = model.predict(n_periods=len(test_data))



   
    # Assuming train_data contains your target variable, and exog_features contains the relevant exogenous features


    st.set_option('deprecation.showPyplotGlobalUse', False)
    import statsmodels.graphics.tsaplots as tsaplots

    import statsmodels.api as sm
    model = sm.tsa.statespace.SARIMAX(endog=train_data, order=(1,0,1), seasonal_order=(0,1,1,12)).fit()


    if model and model.resid.shape[0] > 0:
        fig, ax = plt.subplots(figsize=(15, 8))
        tsaplots.plot_acf(np.squeeze(model.resid), ax=ax)
        ax.set_title('Auto ARIMA Residual Autocorrelation')
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(15, 8))
        tsaplots.plot_pacf(np.squeeze(model.resid), ax=ax)
        ax.set_title('Auto ARIMA Residual Partial Autocorrelation')
        st.pyplot(fig)
    
    hist_data = [model.resid]
    group_labels = ['Residual']
    fig = plt.figure(figsize=(10,5), dpi=100)
    plt.hist(hist_data, color=['orange'], bins=30, alpha=0.7, label=group_labels)
    plt.legend()
    plt.title('Auto ARIMA Residual Histogram')
    st.pyplot(fig)
    
    st.subheader('Predicted Stock Prices')

    arima_model = fit_arima_model(train_data)

    # Show the summary of the ARIMA model
    st.write('ARIMA model summary:')
    st.write(arima_model.summary())

    # Display model summary and diagnostic plots

    st.text(str(arima_model.cov_params()))
    # Make predictions with the ARIMA model
   
    
  
    forecast = model.predict(n_periods=len(train_data))

    # Calculate MSE and RMSE
    mse = mean_squared_error(train_data, forecast)
    rmse = math.sqrt(mse)

    st.write('Mean Squared Error (MSE):', mse)
    st.write('Root Mean Square Error (RMSE):', rmse)
    import matplotlib.pyplot as plt
   

    # Create forecast plot
    plt.figure(figsize=(10, 5))
    plt.plot(test_data, label='Predicted')
    plt.plot(forecast, color='red', label='Actual')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Actual vs. Predicted Values')
    plt.legend()

    # Show plot in Streamlit app
    st.pyplot()
