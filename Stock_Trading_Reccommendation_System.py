#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the necessary libraries
import sqlalchemy
#!pip install pymysql
import pymysql
import pandas as pd
#!pip install yfinance
import yfinance
import warnings
#!pip install mysql-connector-python
#!sudo service mysql start
warnings.filterwarnings("ignore")


# In[2]:


# Create a function to create a schema in the MySQL database for each index you want to use in the system.
import mysql.connector

def schemacreator(index):
    # Establishing a connection
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Ngetich30"
    )
    # Creating a cursor
    mycursor = mydb.cursor()

    # Creating the schema
    mycursor.execute(f"DROP SCHEMA IF EXISTS {index}")

    # Creating a cursor
    mycursor = mydb.cursor()

    # Creating the schema
    mycursor.execute(f"CREATE SCHEMA IF NOT EXISTS {index}")

    # Closing the cursor and connection
    mycursor.close()
    mydb.close()
schemacreator('my_index')


# In[3]:


# Define the indices you want to use in the system.

indices = ['Nasdaq']

# Use the "pd.read_html" function to scrape the Wikipedia page of the index you want to use and store it in a pandas DataFrame.

nasdaq = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')[4]


# In[4]:


# Filter for semiconductor companies in the Consumer Electronics sector
nasdaq = nasdaq[(nasdaq['GICS Sub-Industry'] == 'Semiconductors')].Ticker.to_list()
print(nasdaq)


# In[5]:


# Connect to the MySQL database and use the yfinance library to download the historical stock price data for each ticker, between the specified start and end dates.

# Establishing a connection
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="Ngetich30",
  database="my_index"
)

for symbol in nasdaq:
    mycursor = mydb.cursor()
    mycursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = %s", (symbol,))
    table_exists = mycursor.fetchone()[0]
    if table_exists > 1:
        print("Tables Exist")
    else:
        cursor = mydb.cursor()
        cursor.execute(f"CREATE TABLE {symbol} (date DATE, open FLOAT, high FLOAT, low FLOAT, close FLOAT, adj_close FLOAT, volume BIGINT)")
    df = yfinance.download(symbol,start='2018-01-01',end='2022-01-01')
    df = df.reset_index()
    cursor = mydb.cursor()
    cursor.execute(f"TRUNCATE TABLE {symbol}")
    for i, row in df.iterrows():
        cursor = mydb.cursor()
        sql = f"INSERT INTO {symbol} (date, open, high, low, close, adj_close, volume) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        val = tuple(row)
        cursor.execute(sql, val)
    mydb.commit()


# In[6]:


# Define a function to retrieve the historical stock price data for a specific ticker from the MySQL database.
def get_data_from_database(ticker):
    cursor = mydb.cursor()
    cursor.execute(f"SELECT * FROM {ticker}")
    rows = cursor.fetchall()
    columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    data = pd.DataFrame(rows, columns=columns)
    data.set_index('Date', inplace=True)
    return data


# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Get data for companies to perform EDA
avgo = get_data_from_database('AVGO')
nvda = get_data_from_database('NVDA')
nxpi = get_data_from_database('NXPI')
qcom = get_data_from_database('QCOM')
txn =  get_data_from_database('TXN')


# #Exploratory Data Analysis
# Use *head()* to view the first few rows of data, and *info()* to get information about the data types and number of non-null values for each column.

# In[8]:


print('AVGO')
avgo.head()


# In[9]:


print('NVDA')
nvda.head()


# In[10]:


print('NXPI')
nxpi.head()


# In[11]:


print('QCOM')
qcom.head()


# In[12]:


print('TXN')
txn.head()


# Check for any missing values using the *isnull()* and *sum()* methods

# In[13]:


print('AVGO: ')
avgo.isnull().sum()


# In[14]:


print('NVDA: ')
nvda.isnull().sum()


# In[15]:


print('NXPI: ')
nxpi.isnull().sum()


# In[16]:


print('QCOM: ')
qcom.isnull().sum()


# In[17]:


print('TXN: ')
txn.isnull().sum()


# Generate summary statistics of the numerical columns using the *describe()* method.

# In[18]:


print('AVGO: ')
avgo.describe()


# In[19]:


print('NVDA: ')
nvda.describe()


# In[20]:


print('NXPI: ')
nxpi.describe()


# In[21]:


print('QCOM: ')
qcom.describe()


# In[22]:


print('TXN: ')
txn.describe()


# # Helper Functions for plotting data

# In[23]:


# Define functions for plotting data
def plot_lines(data):
  data.plot(y='Open')
  plt.title('Open Prices over Time')
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.show()


# In[24]:


def visualize_col_relationships(data):
  fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
  data.plot(y='Open', ax=axes[0,0])
  data.plot(y='Close', ax=axes[0,1])
  data.plot(y='Volume', ax=axes[1,0])
  data.plot(y='Adj Close', ax=axes[1,1])
  plt.show()

  data.plot(kind='scatter', x='Volume', y='Adj Close')
  plt.title('Adj Close Prices vs. Volume')
  plt.show()


# In[25]:


def summary_stats(data):
  # Calculate the mean closing price for each year
  # extract year from 'Date' column
  data.index = pd.to_datetime(data.index)
  data['Year'] = data.index.year
  mean_close_by_year = data.groupby('Year')['Close'].mean()
  print(mean_close_by_year)

  # Calculate the daily price change
  data['Price_Change'] = data['Close'] - data['Open']
  return mean_close_by_year


# In[26]:


def plot_trends(data, mcby):
  fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

  # Plot mean closing price by year
  mcby.plot(ax=axes[0])
  axes[0].set_xlabel('Year')
  axes[0].set_ylabel('Mean Closing Price')
  axes[0].set_title('Mean Closing Price by Year')

  # Plot daily price change
  data['Date'] = pd.to_datetime(data.index)
  data.plot(x='Date', y='Price_Change', ax=axes[1])
  axes[1].set_xlabel('Date')
  axes[1].set_ylabel('Price Change')
  axes[1].set_title('Daily Price Change')

  plt.tight_layout()
  plt.show()


# # Create Plots
# *ticker*: **AVGO**
# 
# Create line plots of the various columns over time. Use the *plot()* method of the dataframe.

# In[27]:


plot_lines(avgo)


# create subplots and scatter plots to visualize relationships between columns.

# In[28]:


visualize_col_relationships(avgo)


# Calculate summary statistics for specific time periods and create new columns based on calculations using the existing columns.

# In[29]:


mcby = summary_stats(avgo)


# Plot the data to provide some insight into the trends and fluctuations in the closing price and daily price change over time.

# In[30]:


plot_trends(avgo, mcby)


# *ticker*: **NVDA**

# Create line plots of the various columns over time. Use the *plot()* method of the dataframe.

# In[31]:


plot_lines(nvda)


# create subplots and scatter plots to visualize relationships between columns.

# In[32]:


visualize_col_relationships(nvda)


# Calculate summary statistics for specific time periods and create new columns based on calculations using the existing columns.

# In[33]:


mcby = summary_stats(nvda)


# Plot the data to provide some insight into the trends and fluctuations in the closing price and daily price change over time.

# In[34]:


plot_trends(nvda, mcby)


# *ticker*: **NXPI**

# Create line plots of the various columns over time. Use the *plot()* method of the dataframe.

# In[35]:


plot_lines(nxpi)


# create subplots and scatter plots to visualize relationships between columns.

# In[36]:


visualize_col_relationships(nxpi)


# Calculate summary statistics for specific time periods and create new columns based on calculations using the existing columns.

# In[37]:


mcby = summary_stats(nxpi)


# Plot the data to provide some insight into the trends and fluctuations in the closing price and daily price change over time.

# In[38]:


plot_trends(nxpi, mcby)


# *ticker*: **QCOM**

# Create line plots of the various columns over time. Use the *plot()* method of the dataframe.

# In[39]:


plot_lines(qcom)


# create subplots and scatter plots to visualize relationships between columns.

# In[40]:


visualize_col_relationships(qcom)


# Calculate summary statistics for specific time periods and create new columns based on calculations using the existing columns.

# In[41]:


mcby = summary_stats(qcom)


# Plot the data to provide some insight into the trends and fluctuations in the closing price and daily price change over time.

# In[42]:


plot_trends(qcom, mcby)


# *ticker*: **TXN**

# Create line plots of the various columns over time. Use the *plot()* method of the dataframe.

# In[43]:


plot_lines(txn)


# create subplots and scatter plots to visualize relationships between columns.

# In[44]:


visualize_col_relationships(txn)


# Calculate summary statistics for specific time periods and create new columns based on calculations using the existing columns.

# In[45]:


mcby = summary_stats(txn)


# Plot the data to provide some insight into the trends and fluctuations in the closing price and daily price change over time.

# In[46]:


plot_trends(txn, mcby)


# 
# #Recommendation System
# Function definition to calculate the moving average of a ticker.

# In[47]:


# Define a function to generate a recommendation based on the MACD, RSI, and moving average of a stock 

def moving_average(ticker, window):
    data = get_data_from_database(ticker)
    ma = data['Close'].rolling(window=window).mean()
    return ma


# Function definition for calculating the Relative Strength Index (RSI) for a given stock ticker and a specified window size.

# In[48]:


def rsi(ticker, window):
    data = get_data_from_database(ticker)
    delta = data['Close'].diff()
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    avg_gain = gains.rolling(window=window).mean()
    avg_loss = losses.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1+ rs))
    return rsi


# Function to calculate the Moving Average Convergence Divergence (MACD) for a given stock ticker.

# In[49]:


def macd(ticker, fast=12, slow=26, signal=9):
    data = get_data_from_database(ticker)
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return histogram


# This function takes in a stock ticker symbol and several optional parameters for the moving average, RSI, and MACD calculations.

# In[63]:


def recommendation(ticker, ma_window=20, rsi_window=14, macd_fast=12, macd_slow=26, macd_signal=9):
    data = get_data_from_database(ticker)
    ma = moving_average(ticker, ma_window)
    RSI = rsi(ticker, rsi_window)
    histogram = macd(ticker, macd_fast, macd_slow, macd_signal)
    # Define the buy and sell signals based on the MACD, RSI, and moving average values
    buy_signal = (histogram > 0) & (RSI < 30) & (data['Close'] > ma)
    sell_signal = (histogram < 0) & (RSI > 70) & (data['Close'] < ma)

    # Check if there are any buy or sell signals and generate a recommendation based on the signals
    if buy_signal.any() and not sell_signal.any():
        return "Strong Buy"
    elif buy_signal.any() and sell_signal.any():
        return "Buy"
    elif not buy_signal.any() and sell_signal.any():
        return "Strong Sell"
    elif not buy_signal.any() and not sell_signal.any():
        return "Neutral"
    else:
        return "Error: Could not generate recommendation"


# In[64]:


if __name__ == '__main__':
    """
    This will generate a recommendation for the stock with ticker symbol AAPL using default parameter values
    print it to the console. You can replace 'AAPL' with any other valid stock ticker symbol to generate a recommendation for a different stock.
    """
    for ticker in nasdaq:
        recommendation_ = recommendation(ticker)
        print(f"Recommendation for {ticker}: {recommendation_}")

