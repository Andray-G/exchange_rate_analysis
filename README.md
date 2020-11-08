# Exchange Rate Analysis Program

This program utilizes Python and machine learning techniques to analyze and forecast the currency exchange rates. Using historical data from Yahoo Finance, the script integrates ARIMA and GARCH models to provide accurate predictions, assess market volatility and provides insight into currency trends.

# Features

  * Historical Data Retrieval: Retrieves exchange rate data from Yahoo Finance.

  * ARIMA Model: Forecasts the next day's exchange rate based on historical trends.

  * GARCH Model: Calculates 95% confidence intervals for residual volatility.

  * Error Metrics: Evaluates forecast accuracy using Mean Absolute Percentage Error (MAPE).

  * Visualization: Plots the exchange rate pairing with a 30-day rolling mean and forecasted value.

  * Result Export: Saves forecast results and confidence intervals to a CSV file for further analysis.

# Commonly Analyzed Currency Pairs

This program can analyze any currency pairing supported by Yahoo Finance. <br /> <br /> Examples include:

  * EUR/USD (Euro / US Dollar)
  * GBP/USD (British Pound / US Dollar)
  * USD/JPY (US Dollar / Japanese Yen)
  * AUD/USD (Australian Dollar / US Dollar)
  * USD/CHF (US Dollar / Swiss Franc)
  * USD/CAD (US Dollar / Canadian Dollar)

# Requirements

## The following Python libraries are required:

* yfinance
* pandas
* numpy
* statsmodels
* arch
* matplotlib
* scikit-learn

## Install the dependencies using:

      pip install yfinance pandas numpy statsmodels arch matplotlib scikit-learn

# Usage

### Clone the repository:

     git clone https://github.com/your-username/exchange_rate_analysis.git

### Navigate to the project directory:

      cd exchange_rate_analysis

### Run the program:

     currency_forcast_index.py

### Output

 The program provides:

  * Predicted selected currency exchange rate  for the next day.

  * 95% confidence intervals for volatility.

  * MAPE to evaluate forecast accuracy.

  * Visualization of the exchange rate with a rolling mean and forecasted value.

  * Results saved to euro_index_forecast_results.csv in the current working directory.

### Example Output

      Fetching data for EURUSD=X from 2019-01-01 to 2020-11-06...
      Data fetched successfully.
      Fitting ARIMA model...
      ARIMA forecast for next day: 1.177321
      Fitting GARCH model on residuals...
      95% Confidence Interval: [-0.007200, 0.007200]
      Mean Absolute Percentage Error (MAPE): 0.35%
      Results saved to euro_index_forecast_results.csv.

This program supports any currency pairing, making it versatile for forex traders, analysts, and financial researchers looking to understand exchange rate trends and volatility.
