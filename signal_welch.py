from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.neural_network import MLPClassifier

print(tf.__version__)

if tf.test.gpu_device_name():
    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")



np.random.seed(1234)

fs = 10e3
N = 1e5
amp = 2*np.sqrt(2)
freq = 1234.0
noise_power = 0.001 * fs / 2
time = np.arange(N) / fs
x = amp*np.sin(2*np.pi*freq*time)
x += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)

f, Pxx_den = signal.welch(x, fs, nperseg=1024)
plt.semilogy(f, Pxx_den)
plt.ylim([0.5e-3, 1])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()

np.mean(Pxx_den[256:])

f, Pxx_spec = signal.welch(x, fs, 'flattop', 1024, scaling='spectrum', return_onesided=False)
plt.figure()
plt.semilogy(f, np.sqrt(Pxx_spec))
plt.xlabel('frequency [Hz]')
plt.ylabel('Linear spectrum [V RMS]')
plt.show()

np.sqrt(Pxx_spec.max())

x[int(N//2):int(N//2)+10] *= 50.
f, Pxx_den = signal.welch(x, fs, nperseg=1024)
f_med, Pxx_den_med = signal.welch(x, fs, nperseg=1024, average='median')
plt.semilogy(f, Pxx_den, label='mean')
plt.semilogy(f_med, Pxx_den_med, label='median')
plt.ylim([0.5e-3, 1])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.legend()
plt.show()