import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import os
import warnings

# Suppress warnings to avoid cluttering the output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# Fetch historical currency exchange rate data from Yahoo Finance.
# - ticker: The exchange rate ticker symbol (e.g., "EURUSD=X").
# - start_date: The start date for fetching the data (YYYY-MM-DD).
# - end_date: The end date for fetching the data (YYYY-MM-DD).
# Returns:
# - A pandas Series containing the 'Close' exchange rate values.
def fetch_currency_data(ticker, start_date, end_date):

    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if data.empty:
        raise ValueError(f"No data found for {ticker} in the specified date range.")

    return data["Close"]



# Fit an ARIMA model and forecast exchange rates.
# Parameters:
# - data: A pandas Series containing the historical exchange rate data.
# - order: A tuple specifying the ARIMA model order (p, d, q).
# Returns:
# - The forecasted value and residuals (errors) of the ARIMA model.
def arima_forecast(data, order):

    model = ARIMA(data, order=order)
    fit = model.fit()
    forecast = fit.forecast(steps=1)
    residuals = fit.resid

    # Clean residuals by removing NaN and infinite values
    residuals = residuals[np.isfinite(residuals)]

    if len(residuals) == 0:
        raise ValueError("Residuals contain only NaN or infinite values after cleaning.")

    return forecast, residuals

# Fit a GARCH model to residuals and calculate 95% confidence intervals.
# Parameters:
# - residuals: The residuals from the ARIMA model.
# - scaling_factor: A scaling factor to adjust the residual values (default is 1000).
# Returns:
# - The lower and upper bounds of the 95% confidence interval.
def garch_confidence_intervals(residuals, scaling_factor=1000):

    residuals = residuals[np.isfinite(residuals)] * scaling_factor

    if len(residuals) == 0:
        raise ValueError("Residuals contain only NaN or infinite values after cleaning.")

    model = arch_model(residuals, vol="Garch", p=1, q=1, rescale=False) # Fit a GARCH(1, 1) model
    fit = model.fit(disp="off") # Suppress verbose output
    forecast = fit.forecast(horizon=1)
    variance = forecast.variance.values[-1, 0] # Extract the variance of the forecast

    # Calculate confidence interval bounds
    lower_bound = (-1.96 * np.sqrt(variance)) / scaling_factor
    upper_bound = (1.96 * np.sqrt(variance)) / scaling_factor
    return lower_bound, upper_bound


# Plot the historical exchange rate, rolling mean and forecast value.
# Parameters:
# - data: The historical exchange rate data.
# - rolling_mean: A rolling mean (e.g., 30-day average) of the data.
# - forecast: The forecasted exchange rate value (optional).
def plot_exchange_rate(data, rolling_mean, forecast=None):

    plt.figure(figsize=(12, 6))
    plt.plot(data, label="Actual Exchange Rate", color="blue")
    plt.plot(rolling_mean, label="30-Day Rolling Mean", color="orange")
    if forecast is not None:
        plt.axhline(forecast, color="red", linestyle="--", label="Forecasted Rate")
    plt.title("EUR/USD Exchange Rate")
    plt.xlabel("Date")
    plt.ylabel("Exchange Rate")
    plt.legend()
    plt.grid()
    plt.show()

# Save a pandas DataFrame of results to a CSV file in the specified directory.
# Parameters:
# - results: A pandas DataFrame containing the results to save.
# - output_dir: The directory where the CSV file will be saved.
# - filename: The name of the CSV file (default is "forecast_results.csv").
def save_results_to_csv(results, output_dir, filename="forecast_results.csv"):

    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, filename)

    try:
        results.to_csv(results_path, index=True)
        print(f"Results successfully saved to: {results_path}")
    except Exception as e:
        print(f"Error saving CSV file: {e}")


 # Main function to fetch data, forecast exchange rates, and analyze findings.
def analyze_currency_data():

    ticker = "EURUSD=X" # Exchange rate ticker symbol
    start_date = "2019-01-01" # Start date for data
    end_date = "2020-11-06" # End date for data
    arima_order = (1, 1, 1) # ARIMA model order (p, d, q)
    scaling_factor = 1000  # Scaling factor to avoid large residuals
    output_dir = r"ENTERFILE_DIRECTORY"  # Directory to save CSV files


    os.makedirs(output_dir, exist_ok=True)

    try:
        # Fetch historical data
        data = fetch_currency_data(ticker, start_date, end_date)
        print("Data fetched successfully.")
        data = data.asfreq('B')  # Adjust data to business days
        data = data.dropna()  # Drop any missing data points

        train_data = data[:-1] # Training data (all but the last point)
        test_data = data[-1:] # Testing data (last point only)

        # Ensure test data is 1-dimensional for MAPE calculation
        test_data = test_data.values.flatten()

        # Fit ARIMA and GARCH models
        forecast, residuals = arima_forecast(train_data * scaling_factor, arima_order)
        forecast = forecast.values.flatten() / scaling_factor  # Rescale forecast back
        lower_bound, upper_bound = garch_confidence_intervals(residuals, scaling_factor)
        mape = mean_absolute_percentage_error(test_data, forecast)


        print(f"ARIMA forecast for next day: {forecast[0]:.6f}")
        print(f"95% Confidence Interval: [{lower_bound:.6f}, {upper_bound:.6f}]")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2%}")

        # Calculate rolling mean for visualization
        rolling_mean = data.rolling(window=30).mean()
        plot_exchange_rate(data, rolling_mean, forecast[0])

        # Create results DataFrame
        results = pd.DataFrame({
            "Actual": test_data,
            "Forecast": forecast,
            "Lower Bound": [lower_bound],
            "Upper Bound": [upper_bound]
        }, index=[data.index[-1]])  # Use the last date as index


        print("Results DataFrame:")
        print(results)


        save_results_to_csv(results, output_dir)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    analyze_currency_data()
