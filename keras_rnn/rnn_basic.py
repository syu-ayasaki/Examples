import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import common

def rnn_basic(X_train, Y_train):

    X_train = np.reshape(X_train, X_train.shape + (1,))

    print(X_train.shape)

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.SimpleRNN(units=10, input_shape=X_train.shape[1:]),
            tf.keras.layers.Dense(1)
        ]
    )

    model.compile(optimizer='adam',
                  loss="mean_squared_error",
                  metrics=['mse'])

    model.fit(X_train, Y_train, epochs=100, verbose=3)

    X_test, _ = np.split(X_train, [1], axis=0)
    Y_predict = model.predict(X_train)

    plt.plot(Y_predict)
    plt.plot(Y_train)
    plt.show()


def main():

    time_steps = 16
    batch_size = 1024
    batch_size_per_phase_cycle = 32

    X_train = []
    Y_train = []

    # make training batch
    for i in range(batch_size):

        phase = batch_size_per_phase_cycle * i * 2.0 * np.pi / batch_size

        x_train, next_phase = common.generate_sinwave_1d(time_steps, s_phase=phase, n_var=0.5)
        y_train, next_phase = common.generate_sinwave_1d(1, s_phase=next_phase, n_var=0.0)

        X_train.append(x_train)
        Y_train.append(y_train)

    X_train = np.stack(X_train)     # [batch_size, time_steps]
    Y_train = np.stack(Y_train)     # [batch_size, 1]

    rnn_basic(X_train, Y_train)


if __name__ == '__main__':

    main()