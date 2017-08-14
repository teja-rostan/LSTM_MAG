import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# Prepare data
df = pd.read_csv("fed/fed.csv", index_col='Date', parse_dates=True,
                 date_parser=lambda date: pd.datetime.strptime(date, '%Y%m'))

data = df['1997-08-01':]
data = data.dropna(axis=1, how='any')

# Create lagged dataset
values = data['Wu-Xia Shadow rate']
df = pd.DataFrame(pd.concat([values.shift(1), values], axis=1))
df.columns = ['t', 't+1']

# split into train and test sets
X = df.as_matrix()
train_size = int(len(X) * 0.6)
train, test = X[1:train_size], X[train_size:]

plt.plot(train[:, 0])
plt.plot([None for i in train[:, 0]] + [x for x in test[:, 0]], color='r')
plt.show()

train_X, train_y = train[:, 0], train[:, 1]
test_X, test_y = test[:, 0], test[:, 1]


#####################
# PERSISTANCE MODEL #
#####################


def model_persistence(x):
    return x

# walk-forward validation PERSISTENCE MODEL
predictions = []
for x in test_X:
    yhat = model_persistence(x)
    predictions.append(yhat)
rmse = np.sqrt(mean_squared_error(test_y, predictions))

print('Test RMSE: %.3f' % rmse)

plt.plot(test)
plt.plot(predictions, color='red')
plt.show()


########################
# AUTOREGRESSION MODEL #
########################


# split dataset
train, test = X[1:train_size, 0], X[train_size:, 0]

# train autoregression
model = AR(train)
model_fit = model.fit()
window = model_fit.k_ar
coef = model_fit.params

# walk forward over time steps in test AUTOREGRESSION MODEL
history = train[-window:].tolist()
predictions = []
for obs in test:
    length = len(history)
    lag = history[length-window:length]
    yhat = np.dot(coef, [1] + lag[::-1])
    predictions.append(yhat)
    history.append(obs)
    # print('predicted=%f, expected=%f' % (yhat, obs))
rmse = np.sqrt(mean_squared_error(test, predictions))

print('Test RMSE: %.3f' % rmse)

plt.plot(test)
plt.plot(predictions, color='red')
plt.show()


########################
# MOVING AVERAGE MODEL #
########################


train_resid = (train_y - train_X).tolist()

# model the training set residuals
model = AR(train_resid)
model_fit = model.fit()
window = model_fit.k_ar
coef = model_fit.params

history = train_resid[-window:]
predictions, expected_error = [], []
for t in range(len(test_y)):
    # persistence
    yhat = test_X[t]
    error = test_y[t] - yhat
    expected_error.append(error)

    # predict error
    length = len(history)
    lag = history[length - window:length]
    pred_error = np.dot(coef, [1] + lag[::-1])
    predictions.append(pred_error)
    history.append(error)
    # print('predicted error=%f, expected error=%f' % (pred_error, error))


# Correct Predictions with a Model of Residuals
history = train_resid[-window:]
predictions, expected_error = [], []
for t in range(len(test_y)):
    # persistence
    yhat = test_X[t]
    error = test_y[t] - yhat

    # predict error
    length = len(history)
    lag = history[length - window:length]
    pred_error = np.dot(coef, [1] + lag[::-1])

    # correct the prediction
    predictions.append(pred_error + yhat)
    history.append(error)
    # print('predicted error=%f, expected error=%f' % (pred_error, error))
rmse = np.sqrt(mean_squared_error(test_y, predictions))

print('Test RMSE: %.3f' % rmse)

# plot prediction
plt.plot(test_y)
plt.plot(predictions, color='red')
plt.show()


###############
# ARIMA MODEL #
###############


s = pd.Series(data['Wu-Xia Shadow rate'])

# fit model
model = ARIMA(s, order=(5, 1, 0))
model_fit = model.fit(disp=0)

# summary of fit model
# print(model_fit.summary())

# line plot of residuals
# residuals = pd.DataFrame(model_fit.resid)
# residuals.plot()

# density plot of residuals
# residuals.plot(kind='kde')

# density plot of residuals, using Seaborn kdeplot
# sns.set(color_codes=True)
# sns.kdeplot(np.hstack(residuals.values), shade=True, color="r").set(xlim=(-2, 2))

# plt.show()

# summary stats of residuals
# print(residuals.describe())


###########################################
# Grid Search ARIMA Model Hyperparameters #
###########################################


# evaluate an ARIMA model for a given order (p, d, q)
def evaluate_arima_model(history_arima, test_arima, arima_order):

    # make predictions
    predictions_arima = []
    for obs in test_arima:
        model_arima = ARIMA(history_arima, order=arima_order)
        model_fit_arima = model_arima.fit(disp=0)
        yhat_arima = model_fit_arima.forecast()[0]
        predictions_arima.append(yhat_arima)
        history_arima.append(obs)
    # calculate out of sample error
    rmse_arima = np.sqrt(mean_squared_error(test_arima, predictions_arima))
    return rmse_arima


# evaluate parameters
p = 3
d = 1
q = 0
order = (p, d, q)
print(evaluate_arima_model(train.tolist(), test, order))


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(history_best, test_best, p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order_best = (p, d, q)
                try:
                    rmse_best = evaluate_arima_model(history_best, test_best, order_best)
                    if rmse_best < best_score:
                        best_score, best_cfg = rmse_best, order_best
                    print('ARIMA%s RMSE=%.3f' % (order_best, rmse_best))
                except:
                    continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))
    return best_cfg


# evaluate parameters
p_values = np.arange(10)
d_values = np.arange(10)
q_values = np.arange(10)
best_order = evaluate_models(train.tolist(), test, p_values, d_values, q_values)


# walk-forward validation
history = train.tolist()
predictions = []
for obs in test:
    model = ARIMA(history, order=best_order)
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    history.append(obs)
    # print('predicted=%f, expected=%f' % (yhat, obs))
rmse = np.sqrt(mean_squared_error(test, predictions))

print('Test RMSE: %.3f' % rmse)

# plot forecasts against actual outcomes
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()

