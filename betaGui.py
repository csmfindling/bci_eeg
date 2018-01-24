import sys
import numpy as np
from PyQt4 import QtGui, QtCore
from vispy import app, scene, visuals, gloo
from vispy.color import Colormap, get_colormap, get_colormaps
from scipy.signal import butter, lfilter
from numpy.fft import fft, fftfreq
import time as timelib
import sip
import pickle
from trainingCanvas import *
from functions import fft_welsh

class betaGui(QtGui.QWidget):
    def __init__(self, parent=None):
        super(betaGui, self).__init__()
        QtGui.QWidget.__init__(self)

        screenShape = QtGui.QDesktopWidget().screenGeometry()
        width  = screenShape.width()*4./10
        height = screenShape.height()*4./10
        self.resize(width, height)

        self.button = QtGui.QPushButton('Start training', self)
        self.layout = QtGui.QVBoxLayout(self)
        self.layout.addWidget(self.button, 0)
        self.button.clicked.connect(self.on_pushButton)
        self.canvas = TrainingCanvas(width, height)
        self.canvas.native.setParent(self)
        self.layout.addWidget(self.canvas.native, 1)
        self.training_ongoing = False
        self.training_done    = False
        self.time_training    = 0.
        self.len_training     = 124 # 124 #  120->2min #288->4min, 36->30sec, 144->2min
        self.labels           = np.zeros(self.len_training)
        self.categories       = ['Left', 'Right']
        self.labels[int(self.len_training/4):int(self.len_training/2)] = 1
        self.labels[-int(self.len_training/4):] = 1
        
    def on_pushButton(self):
        self.layout.removeWidget(self.button)
        sip.delete(self.button)
        self.canvas.update_seconds()
        self.time_training    = 0
        self.training_ongoing = True

    def return_status(self):
        return self.training_ongoing
        
    def update_training(self, dt_time):
        if (self.time_training > 1) and (self.canvas.seconds==3):
            self.canvas.update_seconds()
        elif (self.time_training > 2) and (self.canvas.seconds==2):
            self.canvas.update_seconds()
        elif (self.time_training > 3) and (self.canvas.seconds==1):
            self.canvas.update_seconds()
            self.time_training = 0.
        elif self.canvas.seconds == 0 or self.canvas.seconds == -1 and self.time_training > 0:
            idx = int(self.time_training)
            if idx >= self.len_training:
                self.training_done = True
            if not self.training_done:
                cat     = self.categories[int(self.labels[idx])]
                if cat != self.canvas.return_text():
                    self.canvas.update_text(cat)
        self.time_training += dt_time

    def return_fft_comparison(self, data, channel, dt):
        nb_points      = len(data[channel])
        idx            = int((nb_points/4)/1200)*1200
        quartile       = int(nb_points/4)
        spec_left , f  = fft_welsh(np.concatenate((data[channel][:idx], data[channel][2 * quartile:(2 * quartile + idx)])), dt)
        spec_right, f  = fft_welsh(np.concatenate((data[channel][quartile:(quartile + idx)], data[channel][-idx:])), dt)
        freq_interest  = f[(f>6)*(f<40)]
        spec_right_int = spec_right[(f>6)*(f<40)]
        spec_left_int  = spec_left[(f>6)*(f<40)]
        return (spec_right_int - spec_left_int)/(spec_right_int + spec_left_int), freq_interest




    # #spec_left0, f   = fft_welsh(data[channel][:idx], dt) #fft_welsh(np.concatenate((data[channel][:idx], data[channel][2 * quartile:(2 * quartile + idx)])), dt)
    # #spec_left1, f   = fft_welsh(data[channel][2 * quartile:(2 * quartile + idx)], dt)
    # spec_left       = fft_welsh(np.concatenate((data[channel][:idx], data[channel][2 * quartile:(2 * quartile + idx)])), dt) #(spec_left0 + spec_left1)/2.
    # #spec_right0, f  = fft_welsh(data[channel][quartile:(quartile + idx)], dt) #fft_welsh(np.concatenate((data[channel][quartile:(quartile + idx)], data[channel][-idx:])), dt)
    # #spec_right1, f  = fft_welsh(data[channel][-idx:], dt)
    # spec_right      = fft_welsh(np.concatenate((data[channel][quartile:(quartile + idx)], data[channel][-idx:])), dt) #(spec_right0 + spec_right1)/2.


