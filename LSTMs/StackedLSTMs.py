from math import sin, pi, exp
from matplotlib import pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense


# generate damped sine wave in [0,1]
def generate_sequence(length, period, decay):
    return [0.5 + 0.5 * sin(2 * pi * i / period) * exp(-decay * i) for i in range(length)]


# generate input and output pairs of damped sine waves
def generate_examples(length, n_patterns, output):
    X, y = list(), list()
    for _ in range(n_patterns):
        p = np.random.randint(10, 20)
        d = np.random.uniform(0.01, 0.1)
        sequence = generate_sequence(length + output, p, d)
        X.append(sequence[:-output])
        y.append(sequence[-output:])
    X = np.array(X).reshape(n_patterns, length, 1)
    y = np.array(y).reshape(n_patterns, output)
    return X, y


# test problem generation
# X, y = generate_examples(20, 5, 5)
# for i in range(len(X)):
#     plt.plot([x for x in X[i, :, 0]] + [x for x in y[i]], '-o')
# plt.show()


#################
# Stacked LSTMs #
#################


# configure problem
length = 50
output = 5

# define model
model = Sequential()
model.add(LSTM(20, return_sequences=True, input_shape=(length, 1)))
model.add(LSTM(20))
model.add(Dense(output))
model.compile(loss='mae', optimizer='adam')
print(model.summary())

# fit model
X, y = generate_examples(length, 10000, output)
model.fit(X, y, batch_size=10, epochs=1)


# evaluate model
X, y = generate_examples(length, 1000, output)
loss = model.evaluate(X, y, verbose=0)
print('MAE: %f' % loss)


# prediction on new data
X, y = generate_examples(length, 1, output)
yhat = model.predict(X, verbose=0)
plt.plot(y[0], label='y')
plt.plot(yhat[0], label='yhat')
plt.legend()
plt.show()