# import keras
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pandas.plotting import lag_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.gofplots import qqplot
from pandas.plotting import autocorrelation_plot

# Prepare data
df = pd.read_csv("fed/fed.csv")
data = df.loc[df['Date'] >= 199708]
data = data.dropna(axis=1, how='any')

# LAG SCATTER PLOT
# plt.figure()
# lag_plot(data['Wu-Xia Shadow rate'], lag=1)
# plt.show()

# Autocorrelation Function (ACF)
# plot_acf(data['Wu-Xia Shadow rate'])
# plt.show()


# Partial Autocorrelation Function (ACF)
# plot_pacf(data['Wu-Xia Shadow rate'])
# plt.show()


# Augmented Dickey-Fuller test
# adf, pvalue, _, _, critical_vals, _ = adfuller(data['Wu-Xia Shadow rate'])
# print('ADF Statistics: ' + str(np.around(adf, 6)) + '\n'
#       + 'p-value: ' + str(np.around(pvalue, 6)) + '\n'
#       + 'Critical Values:\n'
#       + '\t\t1%: ' + str(np.around(critical_vals['1%'], 3)) + '\n'
#       + '\t\t5%: ' + str(np.around(critical_vals['5%'], 3)) + '\n'
#       + '\t\t10%: ' + str(np.around(critical_vals['10%'], 3)) + '\n')


###################
# Evaluate models #
###################


# # Multiple Train-Test Splits: calculate repeated train-test splits of time series data
# X = data['Wu-Xia Shadow rate'].values
# splits = TimeSeriesSplit(n_splits=3)
# index = 1
# plt.figure(1, figsize=(15, 9))
#
# for train_index, test_index in splits.split(X):
#     train = X[train_index]
#     test = X[test_index]
#
#     print('Observations: %d' % (len(train) + len(test)))
#     print('Training Observations: %d' % (len(train)))
#     print('Testing Observations: %d\n-----' % (len(test)))
#
#     plt.subplot(310 + index)
#     plt.plot(train)
#     plt.plot([None for i in train] + [x for x in test], color='r')
#     index += 1
# plt.show()


# Walk Forward Validation
# X = data['Wu-Xia Shadow rate'].values
# X = X[-100:]
# n_train = 80
# n_records = len(X)
#
# for i in range(n_train, n_records):
#     train, test = X[0:i], X[i:i+1]
#     print('train=%d, test=%d' % (len(train), len(test)))


#####################################
# Persistence Model for Forecasting #
#####################################


# Create lagged dataset
values = data['Wu-Xia Shadow rate']
df = pd.DataFrame(pd.concat([values.shift(1), values], axis=1))
df.columns = ['t', 't+1']

# split into train and test sets
X = df.as_matrix()
train_size = int(len(X) * 0.66)
train, test = X[1:train_size], X[train_size:]

train_X, train_y = train[:, 0], train[:, 1]
test_X, test_y = test[:, 0], test[:, 1]


# persistence model
def model_persistence(x):
    return x


# walk-forward validation
predictions = []
for x in test_X:
    yhat = model_persistence(x)
    predictions.append(yhat)
rmse = np.sqrt(mean_squared_error(test_y, predictions))

print('Test RMSE: %.3f' % rmse)

# # plot predictions and expected results on the test data
# plt.plot(train_y[-50:])
# plt.plot([None for i in train_y[-50:]] + [x for x in test_y])
# plt.plot([None for i in train_y[-50:]] + [x for x in predictions])
# plt.show()


# # calculate residuals from the above persistence model
# residuals = test_y-predictions
# residuals = pd.DataFrame(residuals)

# # plot residuals
# residuals.plot()
# plt.show()
# print(residuals.describe())


# # histograms and density plots
# residuals.hist()
# residuals.plot(kind='kde')
# plt.show()


# # Residual Q-Q and residual autocorrelation plots and autocorrelation plot of residuals as a line plot
# qqplot(residuals, line='r')
# autocorrelation_plot(residuals)
# plot_acf(residuals, lags=20)
# plt.show()


############################################
# Reframe Time Series Forecasting Problems #
############################################


# Classification Framings

# Create lagged dataset
values = data['Wu-Xia Shadow rate']
df = pd.DataFrame(pd.concat([values.shift(1), values], axis=1))
df.columns = ['t', 't+1']


def make_discrete(row):
    if row['t+1'] < 0:
        return 'low'
    elif row['t+1'] > 3:
        return 'high'
    else:
        return 'medium'

# apply the above function to reassign t+1 values
df['t+1'] = df.apply(lambda row: make_discrete(row), axis=1)

# Randomly sample 10 elements from the dataframe
print(df.sample(n=10))

# Time Horizon Framings

# create lagged dataset
values = data['Wu-Xia Shadow rate']
df = pd.DataFrame(pd.concat([values.shift(1), values, values.shift(-1), values.shift(-2)], axis=1))
df.columns = ['t', 't+1', 't+2', 't+3']

print(df.head())





