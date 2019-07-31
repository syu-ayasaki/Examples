import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def generate_sinwave(size = 512,
                     dt = 1.0,
                     freq = 0.1,
                     s_scale = 1,
                     s_bias = 0,
                     init_phase = 0.0,
                     n_mean = 0,
                     n_var = 0.0
                     ):

    time_series = np.arange(size) * dt
    omega = 2.0 * np.pi * freq

    sin_wave = s_scale * np.sin(time_series * omega + init_phase) + s_bias
    noise_series = np.random.normal(n_mean, n_var, size)

    return sin_wave + noise_series



def main():

    N_len = 64
    N_samples = 1024
    N_cycle = 32

    train_data = []
    for i in range(N_samples):
        sin_wave = generate_sinwave(N_len + 1, init_phase=i * 2.0 * np.pi / N_samples * N_cycle, n_var=0.0)
        train_data.append(sin_wave)
    train_data = np.stack(train_data)

    # print(train_data)
    # plt.imshow(train_data)
    # plt.show()

    x_train, y_train = np.split(train_data, [N_len], axis=1)
    x_train = np.stack([x_train]).reshape((N_samples, N_len, 1))
    x_test, _ = np.split(x_train, [1], axis=0)
    print(x_train.shape)


    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.SimpleRNN(units=10, input_shape=(N_len, 1)),
            tf.keras.layers.Dense(1)
        ]
    )

    model.compile(optimizer='adam',
                  loss="mean_squared_error",
                  metrics=['mse'])

    model.fit(x_train, y_train, epochs=100)

    # y_predict = model.predict(x_train)
    # plt.plot(y_predict)
    # plt.plot(y_train)
    # plt.show()

    print(x_test.shape)
    y_predict = model.predict(x_test)
    y_predict_list = [y_predict]

    for i in range(1024):
        y_predict = model.predict(np.reshape(y_predict, (1,1,1)))
        y_predict_list.append(y_predict)
    plt.plot(y_train)



if __name__ == '__main__':

    main()
    exit()


