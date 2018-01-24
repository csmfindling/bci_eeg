import sys
from PyQt4.QtGui import *
from PyQt4.QtCore import *

from pylab import *
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

class plots(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.setWindowTitle("Figures")
        self.layout = QVBoxLayout()

        self.fig  = Figure()
        self.axes = self.fig.add_subplot(111)
 
        self.alpha_x = np.linspace(0,10,10)
        self.alpha_y = np.zeros(len(self.alpha_x))
        self.line,   = self.axes.plot(self.alpha_x, self.alpha_y)
        #self.line1,  = self.axes.plot(self.alpha_x, self.alpha_y)

        self.beta_x_l = np.linspace(0,10,10)
        self.beta_y_l = np.zeros(len(self.beta_x_l))

        self.beta_x_r = np.linspace(0,10,10)
        self.beta_y_r = np.zeros(len(self.beta_x_r))

        self.mov_x = np.linspace(0,10,10)
        self.mov_y = np.zeros(len(self.beta_x_r))

        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)  # the matplotlib canvas

        self.bouton_beta_r = QPushButton("Beta Right")
        self.bouton_beta_r.clicked.connect(self.beta_r_plot)
        self.layout.addWidget(self.bouton_beta_r)

        self.bouton_beta_l = QPushButton("Beta Left")
        self.bouton_beta_l.clicked.connect(self.beta_l_plot)
        self.layout.addWidget(self.bouton_beta_l)

        self.bouton_alpha = QPushButton("Alpha")
        self.bouton_alpha.clicked.connect(self.alpha_plot)
        self.layout.addWidget(self.bouton_alpha)

        self.bouton_mov = QPushButton("Movement")
        self.bouton_mov.clicked.connect(self.mov_plot)
        self.layout.addWidget(self.bouton_mov)

        self.setLayout(self.layout)
        self.show()

    def load_alpha_info(self, freq, alpha):
        self.alpha_x = freq
        self.alpha_y = alpha

    def load_beta_l_info(self, freq, beta):
        self.beta_x_l = freq
        self.beta_y_l = beta

    def load_beta_r_info(self, freq, beta):
        self.beta_x_r = freq
        self.beta_y_r = beta

    def load_mov_info(self, pred, labels):
        self.mov_x = np.arange(len(pred))
        self.mov_y = pred * 2 - 1

    def alpha_plot(self):
        self.line.set_xdata(self.alpha_x)
        self.line.set_ydata(self.alpha_y)
        self.axes.set_ylim([min(self.alpha_y),max(self.alpha_y)])
        self.axes.set_xlim([min(self.alpha_x),max(self.alpha_x)])
        self.canvas.draw()

    def beta_l_plot(self):
        self.line.set_xdata(self.beta_x_l)
        self.line.set_ydata(self.beta_y_l)
        self.axes.set_ylim([min(self.beta_y_l),max(self.beta_y_l)])
        self.axes.set_xlim([min(self.beta_x_l),max(self.beta_x_l)])
        self.canvas.draw()

    def beta_r_plot(self):
        self.line.set_xdata(self.beta_x_r)
        self.line.set_ydata(self.beta_y_r)
        self.axes.set_ylim([min(self.beta_y_r),max(self.beta_y_r)])
        self.axes.set_xlim([min(self.beta_x_r),max(self.beta_x_r)])
        self.canvas.draw()


    def mov_plot(self):
        self.line.set_xdata(self.mov_x)
        self.line.set_ydata(self.mov_y)
        self.axes.set_ylim([-1.1,1.1])
        self.axes.set_xlim([min(self.mov_x),max(self.mov_x)])
        self.canvas.draw()
