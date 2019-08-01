import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import common

def quantized_onehot(value: float,
                     v_max: float,
                     v_min: float,
                     size: int):

    norm_value = (value - v_min) / (v_max - v_min) * size
    index = int(norm_value)
    if index < 0      : index = 0
    if index > size-1 : index = size - 1

    onehot_array = np.zeros(size)
    onehot_array[index] = 1

    # # filter
    # kernel = np.array([0.1,0.2,0.4,0.2,0.1])
    # onehot_array = np.convolve(onehot_array, kernel, mode='same')
    # onehot_array = onehot_array / np.sum(onehot_array)
    # print(onehot_array)

    return onehot_array


def quantized_onehot_series(v_series: np.ndarray,
                            v_max: float,
                            v_min: float,
                            size: int):

    onehot_series = []
    for i in range(len(v_series)):
        onehot_array = quantized_onehot(v_series[i], v_max, v_min, size)
        onehot_series.append(onehot_array)
    onehot_series = np.stack(onehot_series)

    return onehot_series


def rnn_sequence_quantized(X_train, Y_train):

    print(X_train.shape)
    print(Y_train.shape)

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.LSTM(units=100,
                                 input_shape=X_train.shape[1:],
                                 return_sequences=True),
            tf.keras.layers.Dropout(0.5),
            # tf.keras.layers.Dense(Y_train.shape[-1]*10, activation=None),
            tf.keras.layers.Dense(Y_train.shape[-1], activation=None),
            tf.keras.layers.Softmax(axis=-1)
        ]
    )

    model.compile(optimizer='adam',
                  loss = tf.losses.mean_squared_error,
                  # loss= tf.losses.softmax_cross_entropy,
                  metrics=['accuracy'])

    model.fit(X_train, Y_train, epochs=100, verbose=2, batch_size=X_train.shape[0]//8, shuffle=True)

    X_test = np.reshape(X_train[0], (1,)+X_train[0].shape)
    Y_predict = model.predict(X_test)[0]

    fig = plt.figure()  # type: plt.Figure
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.imshow(Y_predict.T)
    ax2.imshow(Y_train[0].T)
    plt.show()


def main():

    ###########
    time_steps = 128
    batch_size = 128
    batch_size_per_phase_cycle = 32
    quantized_size = 32

    X_train = []
    Y_train = []

    freq = 0.01

    for i in range(batch_size):

        init_phase = batch_size_per_phase_cycle * i * 2.0 * np.pi / batch_size
        _, next_phase = common.generate_sinwave_1d(1, s_phase=init_phase)   # get phase after 1 step

        x_train, _ = common.generate_sinwave_1d(time_steps, freq=freq, s_phase=init_phase, n_var=0.0)
        y_train, _ = common.generate_sinwave_1d(time_steps, freq=freq, s_phase=next_phase, n_var=0.0)

        X_train.append(x_train)
        Y_train.append(quantized_onehot_series(y_train, v_max=1, v_min=-1, size=quantized_size))

    X_train = np.stack(X_train)                             # [batch_size, time_steps]
    X_train = np.reshape(X_train, X_train.shape + (1,))     # [batch_size, time_steps, input_channel]
    Y_train = np.stack(Y_train)                             # [batch_size, time_steps, classifications]

    rnn_sequence_quantized(X_train, Y_train)


if __name__ == '__main__':

    main()
    exit()


