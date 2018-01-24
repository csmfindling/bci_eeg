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

class Second(QtGui.QWidget):
    def __init__(self, parent=None):
        super(Second, self).__init__()
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
        self.nb_seconds_label = 5
        self.len_training     = 165 + self.nb_seconds_label # 288->4min, 36->30sec, 144->2min
        self.labels           = np.zeros(self.len_training)
        self.categories       = ['Nothing', 'Left', 'Right']
        count = 0
        while count < (self.len_training - self.nb_seconds_label):
            U = np.random.rand() > .5
            assert(count%(3*self.nb_seconds_label) == 0)   
            self.labels[count:(count+self.nb_seconds_label)] = 0
            count += self.nb_seconds_label
            self.labels[count:(count+self.nb_seconds_label)] = 1 * U + 2 * (1 - U)
            count += self.nb_seconds_label
            # self.labels[count:(count+self.nb_seconds_label)] = 0
            # count += self.nb_seconds_label
            self.labels[count:(count+self.nb_seconds_label)] = 2 * U + 1 * (1 - U)
            count += self.nb_seconds_label
        #print(count)
        assert(count==(self.len_training - self.nb_seconds_label))
        self.labels[count:(count+self.nb_seconds_label)] = 0
        
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