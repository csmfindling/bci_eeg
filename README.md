## Brain computer interface with online eeg processing
Brain Computer Interface based on 8 channels <a href="http://openbci.com/" class="underline">OpenBCI </a>
 EEG

## Goal
Online detection of the alpha and beta rythms from eeg data.

## Requirements
EEGSynth : http://eegsynth.org/

## Repo description
The main script is mainCanvas.py.

Obtaining the alpha and beta waves are obtained with both wavelet transforms and Welch fourier transform.
Both signals are plotted online and the alpha signal is written online in a redis database that can be retrieved
as a control signal.

The wavelet transform is also served to an LSTM (implemented with tensorflow) to predict intention of movements.
