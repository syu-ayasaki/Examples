import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import common

def rnn_sequence(X_train, Y_train):

    X_train = np.reshape(X_train, X_train.shape + (1,))
    Y_train = np.reshape(Y_train, Y_train.shape + (1,))

    print(X_train.shape)
    print(Y_train.shape)

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.SimpleRNN(units=10, input_shape=X_train.shape[1:], return_sequences=True),
            tf.keras.layers.Dense(1)
        ]
    )

    model.compile(optimizer='adam', loss="mean_squared_error", metrics=['mse'])
    model.fit(X_train, Y_train, epochs=100, verbose=2)

    X_test = np.reshape(X_train[0], (1,)+X_train[0].shape)

    Y_predict = model.predict(X_test)[0]
    plt.plot(Y_predict)
    plt.plot(Y_train[0])
    plt.show()


def main():


    ###########
    time_steps = 1024
    batch_size = 128
    batch_size_per_phase_cycle = 32

    X_train_seq = []
    Y_train_seq = []

    freq = 0.01

    for i in range(batch_size):

        phase = batch_size_per_phase_cycle * i * 2.0 * np.pi / batch_size
        _, next_phase = common.generate_sinwave_1d(1, s_phase=phase)   # get phase after 1 step

        x_train_seq, _ = common.generate_sinwave_1d(time_steps, freq=freq, s_phase=phase, n_var=0.1)
        y_train_seq, _ = common.generate_sinwave_1d(time_steps, freq=freq, s_phase=next_phase, n_var=0.0)

        X_train_seq.append(x_train_seq)
        Y_train_seq.append(y_train_seq)

    X_train_seq = np.stack(X_train_seq)     # [batch_size, time_steps]
    Y_train_seq = np.stack(Y_train_seq)     # [batch_size, time_steps]

    plt.plot(X_train_seq[0])
    plt.plot(Y_train_seq[0])
    plt.show()

    rnn_sequence(X_train_seq, Y_train_seq)



if __name__ == '__main__':

    main()
    exit()


