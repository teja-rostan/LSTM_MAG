import pandas as pd
from matplotlib import pyplot as plt


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










