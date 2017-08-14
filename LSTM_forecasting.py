import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
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


#################
# Vanilla LSTMs #
#################


# one hot encode sequence
def one_hot_encode(sequence, n_features):
    encoding = np.zeros((len(sequence), n_features))
    for i, value in enumerate(sequence):
        encoding[i, value] = 1
    return encoding


# decode a one hot encoded string
def one_hot_decode(encoded_seq):
    return np.argmax(encoded_seq, axis=1)


# generate one example for an lstm
def generate_example(length, n_features, out_index):
    sequence = np.random.randint(n_features, size=length)
    encoded = one_hot_encode(sequence, n_features)
    X = encoded.reshape((1, length, n_features))
    y = encoded[out_index].reshape(1, n_features)
    return X, y


# define model
length = 5
n_features = 10
out_index = 2
model = Sequential()
model.add(LSTM(25, input_shape=(length, n_features)))
model.add(Dense(n_features, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())



