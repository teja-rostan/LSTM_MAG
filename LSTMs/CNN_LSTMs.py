import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Conv2D, MaxPooling2D, Flatten


# generate the next frame in the sequence
def next_frame(last_step, last_frame, column):
    # define the scope of the next step
    lower = max(0, last_step-1)
    upper = min(last_frame.shape[0]-1, last_step+1)
    # choose the row index for the next step
    step = np.random.randint(lower, upper+1)
    # copy the prior frame
    frame = last_frame.copy()
    # add the new step
    frame[step, column] = 1
    return frame, step


# generate a sequence of frames of a dot moving across an image
def build_frames(size):
    frames = list()
    # create the first frame
    frame = np.zeros((size, size))
    step = np.random.randint(0, size)
    # decide if we are heading left or right
    right = 1 if np.random.random() < 0.5 else 0
    col = 0 if right else size-1
    frame[step, col] = 1
    frames.append(frame)
    # create all remaining frames
    for i in range(1, size):
        col = i if right else size-1-i
        frame, step = next_frame(step, frame, col)
        frames.append(frame)
    return frames, right


# # generate sequence of frames
# size = 5
# frames, right = build_frames(size)
# # plot all frames
# plt.figure()
# for i in range(size):  # create a grayscale subplot for each frame
#     plt.subplot(1, size, i+1)
#     plt.imshow(frames[i], cmap='Greys')
#     # turn of the scale to make it clearer
#     ax = plt.gca()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# # show the plot
# plt.show()


# generate multiple sequences of frames and reshape for network input
def generate_examples(size, n_patterns):
    X = np.zeros((n_patterns, size, size, size))
    y = []
    for i in range(n_patterns):
        frames, right = build_frames(size)
        X[i] = frames
        y.append(right)
    # resize as [samples, timesteps, width, height, channels]
    X = X.reshape(n_patterns, size, size, size, 1)
    y = np.array(y).reshape(n_patterns, 1)
    return X, y

# configure problem
size = 50

# define the model
model = Sequential()
model.add(TimeDistributed(Conv2D(2, (2, 2), activation='relu'), input_shape=(None, size, size, 1)))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())


# fit model
X, y = generate_examples(size, 5000)
model.fit(X, y, batch_size=32, epochs=1)


# evaluate model
X, y = generate_examples(size, 100)
loss, acc = model.evaluate(X, y, verbose=0)
print('loss: %f, acc: %f' % (loss, acc*100))


# prediction on new data
X, y = generate_examples(size, 1)
yhat = model.predict_classes(X, verbose=0)
expected = "Right" if y[0]==1 else "Left"
predicted = "Right" if yhat[0]==1 else "Left"
print('Expected: %s, Predicted: %s' % (expected, predicted))
