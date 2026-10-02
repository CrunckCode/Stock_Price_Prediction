# Stock Price Prediction Using Linear Regression (Infosys)

A small linear regression that predicts Infosys (NSE: INFY) closing price three trading days ahead from the day's open, high and low prices.

## Data
`infy.csv`: 251 trading days of INFY data from the NSE starting 13 April 2018 (open, high, low, close, volume, turnover and other fields). Only Open, High, Low and Close are used.

## Method
1. Select Open, High, Low and Close prices.
2. Create the label as the closing price shifted by 3 days (`Close.shift(-3)`).
3. Hold out the last 3 rows as "recent" inputs to forecast, and use the rest for modelling.
4. Random 80/20 train/test split, fit `LinearRegression`, report the test R-squared ("confidence"), and predict the 3 forecast rows.

## Results (re-run)
The reported test R-squared depends heavily on the random split, because there is no fixed seed: over 20 different seeds it ranged from 0.75 to 0.99 with an average of 0.92. Open, high, low and close are strongly correlated with the closing price three days later, so a high R-squared is expected.

## How to run
```
pip install numpy pandas scikit-learn
python stock_market.py
```

## Limitations
- Randomly shuffling a time series puts future days in the training set and past days in the test set, so the R-squared overstates how well the model would forecast truly unseen future prices. A chronological split (train on earlier days, test on later days) is the correct evaluation.
- Price levels are highly autocorrelated; a model that simply predicts "price stays about the same" would also score a high R-squared. Compare against that baseline before drawing conclusions.
- One stock, one year, three input features. This is a learning exercise, not a trading model.
