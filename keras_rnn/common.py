import numpy as np

def generate_sinwave_1d(size    : int   = 512,  # array size
                        dt      : float = 1.0,  # delta time [s]
                        freq    : float = 0.1,  # frequency [Hz]
                        s_scale : float = 1.0,  # signal scaling
                        s_bias  : float = 0.0,  # signal bias
                        s_phase : float = 0.0,  # signal phase
                        n_mean  : float = 0.0,  # noise mean
                        n_var   : float = 0.0   # noise variance):
                        ):

    time_series = np.arange(size) * dt
    omega = 2.0 * np.pi * freq

    sin_wave = s_scale * np.sin(time_series * omega + s_phase) + s_bias
    noise_series = np.random.normal(n_mean, n_var, size)

    sin_wave_with_noise = sin_wave + noise_series
    next_phase = (size*dt) * omega + s_phase

    return sin_wave_with_noise, next_phase