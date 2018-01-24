import sys
import numpy as np
from PyQt4 import QtGui, QtCore
from vispy import app, scene, visuals, gloo
from vispy.color import Colormap, get_colormap, get_colormaps
from scipy.signal import butter, lfilter, detrend
from numpy.fft import fft, fftfreq
import time as timelib
import sip
import pickle

def butter_bandpass(lowcut, highcut, fs, order=9):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=9):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass(lowcut, fs, order=9):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='low')
    return b, a

def butter_lowpass_filter(data, lowcut, fs, order=9):
    b, a = butter_lowpass(lowcut, fs, order=order)
    y    = lfilter(b, a, data)
    return y

def get_dilated_morlet(M, w=5.0, s=1., dt=.00625):
    x       = np.linspace(-M/2,M/2,M+1) * dt #np.linspace(-2*np.pi, 2*np.pi, M) * s
    output  = np.exp(1j * w * x * s)
    output -= np.exp(-0.5 * (w**2))
    output *= np.exp(-0.5 * ((x*s)**2)) * np.pi**(-0.25)
    output *= np.power(1 + np.exp(-w**2) - 2 * np.exp(-3./4 * w**2) , -1./2)
    output *= s
    return [x, output]

def wavelet_transform(Y, dt):
    assert(len(Y)==100)
    nb_filter   = 40
    s_list_f    = 2**(np.arange(nb_filter)/10.)
    w_list_f    = np.ones(nb_filter)*2.*2*np.pi
    results     = np.zeros([nb_filter])
    for i in range(len(w_list_f)):
        x, morlet_d = get_dilated_morlet(M=99, w=w_list_f[i],s=s_list_f[i], dt=dt)
        results[i]  = np.abs(np.convolve(morlet_d, Y, mode='valid'))
    return results, w_list_f*s_list_f/(2*np.pi)

def detect_alpha_freq(Y, dt, borne=[7,15]):
    assert(len(Y) == 1200)
    len_cycle         = 240
    iter_cycle        = 60
    nb_cycles         = 17
    bins_center       = np.arange(0, nb_cycles) * iter_cycle + len_cycle/2
    freq_fourier      = fftfreq(len_cycle, dt)
    freq_fourier      = freq_fourier[:int(len_cycle/2)]
    fourier_trans_l   = np.zeros([nb_cycles, len(freq_fourier)])
    for cycle_idx in range(nb_cycles):
        fourier_trans_l[cycle_idx] = np.abs(np.fft.fft(Y[int(bins_center[cycle_idx] - len_cycle/2):int(bins_center[cycle_idx] + len_cycle/2)]))[:int(len_cycle/2)] # 500ms
    fourier_trans_av = np.mean(fourier_trans_l, axis=0)
    index_freq       = np.where((freq_fourier > borne[0]) & (freq_fourier < borne[1]))[0]
    return fourier_trans_av, freq_fourier, index_freq[np.argmax(fourier_trans_av[index_freq])]

def fft_welsh(Y, dt):
    assert(len(Y)%1200==0)
    len_cycle         = 240
    iter_cycle        = 60
    nb_cycles         = int((len(Y) - len_cycle)/iter_cycle + 1) #17
    bins_center       = np.arange(0, nb_cycles) * iter_cycle + len_cycle/2
    freq_fourier      = fftfreq(len_cycle, dt)
    freq_fourier      = freq_fourier[:int(len_cycle/2)]
    fourier_trans_l   = np.zeros([int(nb_cycles), len(freq_fourier)])
    for cycle_idx in range(int(nb_cycles)):
        fourier_trans_l[cycle_idx] = np.abs(np.fft.fft(detrend(Y[int(bins_center[cycle_idx] - len_cycle/2):int(bins_center[cycle_idx] + len_cycle/2)])))[:int(len_cycle/2)] # 500ms
    fourier_trans_av = np.mean(fourier_trans_l, axis=0)
    return fourier_trans_av, freq_fourier




