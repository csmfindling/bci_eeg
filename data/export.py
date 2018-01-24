import pickle
import sys
import numpy as np
from scipy.signal import butter, lfilter, detrend
from numpy.fft import fft, fftfreq
import time as timelib
import pickle

# functions
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

# data
ref   = pickle.load(open('python_data/rawref.pkl'))
close = pickle.load(open('python_data/rawclose.pkl'))
eye   = pickle.load(open('python_data/rawblink.pkl'))


# parameters
dt            = 0.004 # sampling frequency is of 1/0.004 = 250Hz

# write in csv for reference = 30sec
nb_points     = int(30 * 1./0.004)
data          = ref[:, -nb_points:]
y             = butter_lowpass_filter(data, 30., 1./dt) # low pass filter
y             = (y - np.mean(y))/np.std(y) # normalize

concat        = np.concatenate((data[:-1], y[:-1]), axis=0)

import csv
with open('csv_data/reference.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['eye', 'channel 2', 'channel 3', 'alpha', 'eye filtered', 'channel 2 filtered', 'channel 3 filtered', 'alpha filtered'])
    for idx in range(concat.shape[-1]):
    	spamwriter.writerow(concat[:,idx])

# write in csv for close_eye = 30 sec
nb_points     = int(30 * 1./0.004)
data          = close[:, -nb_points:]
y             = butter_lowpass_filter(data, 30., 1./dt) # low pass filter
y             = (y - np.mean(y))/np.std(y) # normalize

concat        = np.concatenate((data[:-1], y[:-1]), axis=0)

import csv
with open('csv_data/close_eye.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['eye', 'channel 2', 'channel 3', 'alpha', 'eye filtered', 'channel 2 filtered', 'channel 3 filtered', 'alpha filtered'])
    for idx in range(concat.shape[-1]):
    	spamwriter.writerow(concat[:,idx])

# write in csv for close_eye = 30 sec
nb_points     = int(30 * 1./0.004)
data          = eye[:, -nb_points:]
y             = butter_lowpass_filter(data, 30., 1./dt) # low pass filter
y             = (y - np.mean(y))/np.std(y) # normalize

concat        = np.concatenate((data[:-1], y[:-1]), axis=0)

import csv
with open('csv_data/blinks.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['eye', 'channel 2', 'channel 3', 'alpha', 'eye filtered', 'channel 2 filtered', 'channel 3 filtered', 'alpha filtered'])
    for idx in range(concat.shape[-1]):
    	spamwriter.writerow(concat[:,idx])





