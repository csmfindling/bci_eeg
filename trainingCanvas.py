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

class TrainingCanvas(app.Canvas):
    def __init__(self, height, width):
        app.Canvas.__init__(self, title='Glyphs', keys='interactive', size=(50,50))
        self.text = visuals.TextVisual('', bold=True)
        self.font_size = 14
        self.seconds   = -1
        self.text.text = 'Waiting to start training...'
        self.text.font_size = self.font_size
        self.text.pos = height/2 - 2, width/2 - 2#260, 140
        self.update()

    def on_draw(self, event):
        gloo.clear(color='white')
        gloo.set_viewport(0, 0, *self.physical_size)
        self.text.draw()

    def on_resize(self, event):
        # Set canvas viewport and reconfigure visual transforms to match.
        vp = (0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*vp)
        self.text.transforms.configure(canvas=self, viewport=vp)

    def update_seconds(self):
        if self.seconds == -1:
            self.seconds = 3
        else:
            self.seconds -= 1
        self.text.text = 'Training Starting in {0}...'.format(self.seconds)
        self.update()

    def update_text(self, text_s):
        self.text.text = text_s
        self.update()

    def return_text(self):
        return self.text.text