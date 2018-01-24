# 1cm 5min
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
from trainingGui import *
from functions import *
import glob
from makehdf5 import makehdf5
from tf_lstm import lstm
from waveletTransform import wavelet_transform as wavelet_trans
import os
from serialRedis import redis_serial
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from betaGui import betaGui
from plots import plots as plots_info

get_colormaps()

class RawPlot(scene.SceneCanvas):

    def __init__(self, time, pos, dt, apply_redis, size = (1400,700)):
        """
        pos is a Matrix  (nChan * Time)
        """
        scene.SceneCanvas.__init__(self, keys="interactive", size=size)
        self.unfreeze()                                    # allow the creation of new attribute to the clas
        self.grid = self.central_widget.add_grid()         #layout definition
        self.training_window = Second()
        # Display parameters
        self.raw_wind_size   = 1200
        self.alpha_wind_size = 1200
        self.raw_interline   = 10
        self.M               = 50
        self.nch             = pos.shape[0]
        cm                   = get_colormap('autumn')
        self.sel             = 0
        # self.alpha_freq      = 20
        self.dt              = dt
        self.pos             = pos[:,-self.raw_wind_size:]
        self.plot_           = butter_lowpass_filter(self.pos, 30., 1./dt)
        self.plot_           = self.plot_[:, 50:]
        self.plot_           = 5 * np.transpose((self.plot_.T - np.mean(self.plot_, axis=1))/(np.std(self.plot_, axis=1)))

        self.starting_sample  = -1
        self.end_sample       = -1
        self.training_type    = ''
        self.motor_canals     = [1,2,3]#[3,4,1] # left, right, ref
        self.lb_motor         = 13
        self.ub_motor         = 30
        self.left_canal       = 3
        self.right_canal      = 2
        self.ref_motor        = 1
        self.lstm             = lstm()
        self.c_state          = np.zeros([1,self.lstm.n_hidden])
        self.m_state          = np.zeros([1,self.lstm.n_hidden])
        self.wt               = wavelet_trans()
        self.prediction       = np.array([0.])
        self.smoothed_preds   = np.array([0.])
        self.benchmark        = np.array([0.])
        self.smoothing_param  = .3
        self.apply_redis      = apply_redis
        self.movering_av      = .1
        self.plots_info       = plots_info()

        if apply_redis:
            self.serial_comm = redis_serial()

        # subj_idx
        self.subj_idx = input("Enter subject name (-1 for no name): ")
        if self.subj_idx == -1:
            self.return_subjidx()

        # beta_gui
        self.beta_gui = betaGui()

        # create folder
        if not os.path.isdir("data/subj{0}".format(self.subj_idx)):
            os.makedirs("data/subj{0}".format(self.subj_idx))

        # RAW DATA DISPLAY
        self.view_raw = scene.widgets.ViewBox(parent=self.scene)
        self.view_raw.camera = 'panzoom'
        self.grid.add_widget(self.view_raw,0,0,col_span=3)
        self.lines = []
        for i in np.arange(self.nch):
            line = scene.visuals.Line(pos=np.array([np.arange(len(self.plot_[i,:])),self.plot_[i,:]+i*self.raw_interline]).T, parent=self.view_raw.scene, color = cm[float(i)/self.nch], width=1)
            self.lines.append(line)
        self.lines[self.sel].set_data(width = 4) # set the color of the selected channel
        self.view_raw.camera.set_range()

        # SPECTRE DISPLAY
        self.view_spectre        = scene.widgets.ViewBox(parent=self.scene)
        self.view_spectre.camera = 'panzoom'
        self.grid.add_widget(self.view_spectre, 1, 0)
        last_index                        = client.getHeader().nSamples
        assert(last_index > self.alpha_wind_size)
        pos_alpha                         = client.getData([(last_index-self.alpha_wind_size),(last_index-1)])[:,canals].T
        Y                                 = np.array(pos_alpha[self.sel,:])
        spec, self.f, self.alpha_index    = detect_alpha_freq(Y, dt, borne=[7,15])
        log_spec                          = np.log(spec)
        self.line_spectre                 = scene.visuals.Line(pos=np.concatenate([[np.arange(len(log_spec))], [log_spec]]).T, parent=self.view_spectre.scene, width=2)
        self.line_alpha_freq              = scene.visuals.Line(pos=np.array([[self.alpha_index-1, np.min(log_spec)], [self.alpha_index-1, np.max(log_spec)]]), parent=self.view_spectre.scene, width=2)
        self.view_spectre.camera.set_range()

        data_filtered   = butter_bandpass_filter(self.pos[self.sel,:], 5, 40, 1./dt, order=9)
        
        # WAVELET TRANSFORM
        self.view_waveletalpha = scene.widgets.ViewBox(parent=self.scene)
        self.view_waveletalpha.camera = 'panzoom'
        self.grid.add_widget(self.view_waveletalpha,1,2)
        modulus, freqencies     = wavelet_transform(data_filtered[-100:], dt)
        alpha_wavelet           = np.mean(modulus[(freqencies>(self.f[self.alpha_index]-1))*(freqencies<(self.f[self.alpha_index]+1))])
        self.alpha_hist_wavelet = np.array([0,alpha_wavelet])
        self.noise_wavelet      = np.mean(modulus[(freqencies>1)*(freqencies<4)])
        self.mean_noise           = self.noise_wavelet
        self.noise_hist_wavelet   = np.array([0,self.noise_wavelet])
        self.alpha_hist_wavelet_f = np.array([0,alpha_wavelet])
        self.line_alpha_wavelet   = scene.visuals.Line(pos=np.concatenate([[np.arange(len(self.alpha_hist_wavelet))], [self.alpha_hist_wavelet]]).T, parent=self.view_waveletalpha.scene, width=2, color = '#00ff00') # green
        self.line_noise_wavelet   = scene.visuals.Line(pos=np.concatenate([[np.arange(len(self.noise_hist_wavelet))], [self.noise_hist_wavelet]]).T, parent=self.view_waveletalpha.scene, width=2, color = '#0000ff') # blue
        self.line_noise_wavelet_f = scene.visuals.Line(pos=np.concatenate([[np.arange(len(self.alpha_hist_wavelet_f))], [self.alpha_hist_wavelet_f]]).T, parent=self.view_waveletalpha.scene, width=2, color = '#ff0000') # red
        self.view_waveletalpha.camera.set_range()

        # BETA DISPLAY
        self.view_alpha         = scene.widgets.ViewBox(parent=self.scene)
        self.view_alpha.camera  = 'panzoom'
        self.grid.add_widget(self.view_alpha,1,1)
        modulus_l, freqencies_l = wavelet_transform(self.pos[self.left_canal,:][-100:], dt)
        beta_freq_l             = np.mean(modulus_l[(freqencies_l>self.lb_motor)*(freqencies_l<self.ub_motor)])
        modulus_r, freqencies_r = wavelet_transform(self.pos[self.right_canal,:][-100:], dt)
        beta_freq_r             = np.mean(modulus_r[(freqencies_r>self.lb_motor)*(freqencies_r<self.ub_motor)])
        self.beta_hist_l        = np.array([0,beta_freq_l])
        self.beta_hist_r        = np.array([0,beta_freq_r])
        self.line_beta_hist_l   = scene.visuals.Line(pos=np.concatenate([[np.arange(len(self.beta_hist_l))], [self.beta_hist_l]]).T, parent=self.view_alpha.scene, width=2, color = '#00ff00') # green
        #self.line_beta_hist_r   = scene.visuals.Line(pos=np.concatenate([[np.arange(len(self.beta_hist_r))], [self.beta_hist_r]]).T, parent=self.view_alpha.scene, width=2, color = '#0000ff') # blue
        self.view_alpha.camera.set_range()


        @self.events.mouse_press.connect
        def on_mouse_press(event):
            # Find the cursor position in the windows coordinate
            tr = self.scene.node_transform(self.lines[0])
            x, y = tr.map(event.pos)[:2]
            self.pressxy = (x,y)
            if y > tr.map(self.view_raw.size)[1]: # be sure that there is no selection change during the modification of the others viewboxes
                self.sel = np.argmin(np.abs(np.arange(self.nch)*self.raw_interline + np.mean(self.plot_,1) - y))  # add signal because here we take only the reference 0 for the click y coord
                for i in range(len(self.lines)): # reset the color of the others
                    self.lines[i].set_data(color=cm[float(i)/self.nch], width = 1)
                self.lines[self.sel].set_data(width = 4) # set the color of the selected channel
            self.update_alpha_freq()

        @self.events.key_press.connect
        def on_key_press(event):
            if event.text=='m':
                self.training_type = 'movements'
                self.training_window.show()
            elif event.text=='i':
                self.training_type = 'intentions'
                self.training_window.show()
            elif event.text=='b':
                self.beta_gui.show()
            elif event.text=='s':
                self.save()

    def save(self):
        data = client.getData().T
        if not os.path.isdir("data/subj{0}".format(self.subj_idx)):
            os.makedirs("data/subj{0}".format(self.subj_idx))
        pickle.dump(data, open('data/subj{0}/rawdata.pkl'.format(self.subj_idx), 'wb'), protocol=2)
        print('saved all data')

    def set_raw_data(self, time, pos):
        self.pos   = pos[:,-self.raw_wind_size:]
        self.plot_ = butter_lowpass_filter(self.pos, 30., 1./dt)
        self.plot_ = self.plot_[:, 50:]
        self.plot_ = 5*np.transpose((self.plot_.T - np.mean(self.plot_, axis=1))/(np.std(self.plot_, axis=1)))
        for i in range(len(self.lines)):
            self.lines[i].set_data(np.array([np.arange(len(self.plot_[i,:])),self.plot_[i,:]+i*self.raw_interline]).T)
        return self

    def save_training_data(self, begin, index):
        if begin:
            self.starting_sample = index
        if not begin:
            self.end_sample = index
            self.data       = client.getData([self.starting_sample - 1, self.end_sample - 1])[:,self.motor_canals].T
            if not os.path.isdir("data/subj{0}".format(self.subj_idx)):
                os.makedirs("data/subj{0}".format(self.subj_idx))
            pickle.dump([self.data, self.training_window.labels], open('data/subj{1}/trainingdata_{0}.pkl'.format(self.training_type, self.subj_idx), 'wb'), protocol=2)

    def train(self):
        n_total_rightleft                 = makehdf5(self.data, self.training_window.labels, self.subj_idx, self.M, self.training_type)
        self.prediction, self.test_labels = self.lstm.train(n_total_rightleft, self.subj_idx, self.training_type)
        self.plots_info.load_mov_info(self.prediction, self.test_labels)

    def return_subjidx(self):
        files = glob.glob('data/subj*/*data*.pkl')
        if len(files)!=0:
            f               = files[-1]
            last_subj_saved = int(files[-1].split('/')[1][4:])
            self.subj_idx   = last_subj_saved + 1 
        else:
            self.subj_idx   = 1
        if not os.path.isdir("data/subj{0}".format(self.subj_idx)):
            os.makedirs("data/subj{0}".format(self.subj_idx))

    def update_alpha(self):
        # Filter
        data_filtered = butter_bandpass_filter(self.pos[self.sel,:], 5, 40, 1./dt, order=9)

        # FOURIER TRANSFORM
        spec, f         = fft_welsh(self.pos[self.sel,:], dt)
        log_spec        = np.log(spec)
        self.line_spectre.set_data(pos=np.concatenate([[np.arange(len(log_spec))], [log_spec]]).T)
        # alpha           = (spec[self.alpha_index - 1] + spec[self.alpha_index] + spec[self.alpha_index + 1])/3.
        # self.alpha_hist = np.append(self.alpha_hist, alpha)
        # alpha_plot      = self.alpha_hist[-100:] # self.alpha_hist[-self.alpha_wind_size:] #self.alpha_hist[-self.alpha_wind_size:]
        # self.noise      = spec[1:5]
        # self.noise      = np.mean(self.noise)
        # self.line_alpha.set_data(pos=np.concatenate([[np.arange(len(alpha_plot))], [alpha_plot]]).T)
        # self.noise_hist = np.append(self.noise_hist, self.noise)
        # noise_plot      = self.noise_hist[-100:]
        # self.line_noise.set_data(pos=np.concatenate([[np.arange(len(noise_plot))], [noise_plot]]).T)
        # maxi = max(alpha_plot.max(), noise_plot.max())
        # mini = min(alpha_plot.min(), noise_plot.min())
        # self.view_alpha.camera.set_range(x=[0,alpha_plot.shape[0]], y=[mini, maxi]) # set camera range

        # ALPHA WAVELET TRANSFORM
        modulus, freqencies     = wavelet_transform(data_filtered[-100:], dt)
        alpha_wavelet           = np.mean(modulus[(freqencies > (self.f[self.alpha_index] - 1.5)) * (freqencies < (self.f[self.alpha_index] + 1.5))]) #np.mean(modulus[(freqencies > 8) * (freqencies < 13)]) #
        self.alpha_hist_wavelet = np.append(self.alpha_hist_wavelet, alpha_wavelet)
        self.noise_wavelet      = np.mean(modulus[(freqencies>1)*(freqencies<4)])
        self.mean_noise         = (len(self.noise_hist_wavelet)-1)/len(self.noise_hist_wavelet) * self.mean_noise + self.noise_wavelet/len(self.noise_hist_wavelet)
        self.noise_hist_wavelet = np.append(self.noise_hist_wavelet, self.noise_wavelet) # truc en vert en bas à droite
        self.line_alpha_wavelet.set_data(pos=np.concatenate([[np.arange(len(self.alpha_hist_wavelet[-100:]))], [self.alpha_hist_wavelet[-100:]]]).T)
        self.line_noise_wavelet.set_data(pos=np.concatenate([[np.arange(len(self.noise_hist_wavelet[-100:]))], [self.noise_hist_wavelet[-100:]]]).T)
        alpha_movering_av         = alpha_wavelet * self.movering_av + (1 - self.movering_av) * self.alpha_hist_wavelet_f[-1]
        self.alpha_hist_wavelet_f = np.append(self.alpha_hist_wavelet_f, self.alpha_hist_wavelet_f[-1] * (self.noise_wavelet > self.mean_noise) + alpha_movering_av * (self.noise_wavelet < self.mean_noise)) # truc en rouge en bas à droite
        self.line_noise_wavelet_f.set_data(pos=np.concatenate([[np.arange(len(self.alpha_hist_wavelet_f[-100:]))], [self.alpha_hist_wavelet_f[-100:]]]).T)
        maxi = max(self.alpha_hist_wavelet[-100:].max(), self.noise_hist_wavelet[-100:].max())#, self.alpha_hist_wavelet_f[:-100].max())
        mini = min(self.alpha_hist_wavelet[-100:].min(), self.noise_hist_wavelet[-100:].min())#, self.alpha_hist_wavelet_f[:-100].min())
        self.view_waveletalpha.camera.set_range(x=[0,self.alpha_hist_wavelet[-100:].shape[0]], y=[mini, maxi]) # set camera range
        if self.apply_redis:
            self.serial_comm.write(self.alpha_hist_wavelet_f[-1])

        # BETA WAVELET TRANSFORM
        if not self.lstm.learning_done:
            modulus_l, freqencies_l = wavelet_transform(self.pos[self.left_canal,:][-100:], dt)
            beta_freq_l             = np.mean(modulus_l[(freqencies_l>self.lb_motor)*(freqencies_l<self.ub_motor)])
            modulus_r, freqencies_r = wavelet_transform(self.pos[self.right_canal,:][-100:], dt)
            beta_freq_r             = np.mean(modulus_r[(freqencies_r>self.lb_motor)*(freqencies_r<self.ub_motor)])
            # print(beta_freq_l)
            # print(beta_freq_r)
            # self.beta_hist_l        = np.array([0,beta_freq_l])
            # self.beta_hist_r        = np.array([0,beta_freq_r])
            # beta_wavelet           = np.mean(modulus[(freqencies > 25) * (freqencies < 35)])
            self.beta_hist_l         = np.append(self.beta_hist_l, beta_freq_l - beta_freq_r)
            self.beta_hist_r         = np.append(self.beta_hist_r, beta_freq_r)
            self.beta_hist_l         = self.beta_hist_l[-100:]
            self.beta_hist_r         = self.beta_hist_r[-100:]
            # print(beta_freq_r)
            # print(beta_freq_l)
            # self.line_alpha.set_data(pos=np.concatenate([[np.arange(len(self.beta_hist[-100:]))], [self.beta_hist[-100:]]]).T)
            # self.line_noise.set_data(pos=np.concatenate([[np.arange(len(self.noise_hist_wavelet[-100:]))], [self.noise_hist_wavelet[-100:]]]).T)       
            # beta_movering_av = beta_wavelet * movering_av + (1 - movering_av) * self.beta_hist_f[-1]
            # self.beta_hist_f = np.append(self.beta_hist_f, self.beta_hist_f[-1] * (self.noise_wavelet > self.mean_noise) + beta_movering_av * (self.noise_wavelet < self.mean_noise))
            self.line_beta_hist_l.set_data(pos=np.concatenate([[np.arange(len(self.beta_hist_l[-100:]))], [self.beta_hist_l[-100:]]]).T)
            #self.line_beta_hist_r.set_data(pos=np.concatenate([[np.arange(len(self.beta_hist_r[-100:]))], [self.beta_hist_r[-100:]]]).T)
            maxi = self.beta_hist_l[-400:].max() #max(self.beta_hist_l[-100:].max(), self.beta_hist_r[-100:].max())#, self.alpha_hist_wavelet_f[:-100].max())
            mini = self.beta_hist_l[-400:].min() #min(self.beta_hist_l[-100:].min(), self.beta_hist_r[-100:].min())#, self.alpha_hist_wavelet_f[:-100].min())
            self.view_alpha.camera.set_range(x=[0,self.beta_hist_l[-100:].shape[0]], y=[mini, maxi]) # set camera range
        else:
            self.beta_hist             = np.array([0])
            self.beta_hist_f           = np.array([0])
            self.benchmark             = np.append(self.benchmark, 0.)
            datapoint                  = np.asarray([[self.wt.transform(pos[self.motor_canals[:-1],-self.M:] - pos[self.motor_canals[-1],-self.M:]).ravel()]])
            pred, lstm_tuple           = self.lstm.evaluate(self.c_state, self.m_state, datapoint)
            self.c_state, self.m_state = lstm_tuple
            self.prediction            = np.append(self.prediction, (pred[0] > .5) * 2 - 1)
            self.smoothed_preds        = np.append(self.smoothed_preds, (1 - self.smoothing_param) * self.smoothed_preds[-1] + self.smoothing_param * self.prediction[-1])
            self.line_beta_hist_l.set_data(pos=np.concatenate([[np.arange(len(self.smoothed_preds[-100:]))], [self.smoothed_preds[-100:]]]).T, color = '#0000ff')
            self.line_beta_hist_r.set_data(pos=np.concatenate([[np.arange(len(self.benchmark[-100:]))], [self.benchmark[-100:]]]).T, width=.5, color = '#ffffff')
            self.view_alpha.camera.set_range(x=[0,self.benchmark[-100:].shape[0]], y=[-1, 1]) # set camera range
        return self

    def update_alpha_freq(self):
        last_index  = client.getHeader().nSamples
        pos_alpha   = client.getData([(last_index-self.alpha_wind_size),(last_index-1)])[:,canals].T
        Y           = np.array(pos_alpha[self.sel,:])
        spec, self.freq, self.alpha_index = detect_alpha_freq(Y, dt, borne= [7, 15])
        # print('alpha index is {0}'.format(self.alpha_index))
        alpha           = spec[self.alpha_index]
        self.alpha_hist = np.array([0,alpha])
        self.alpha_hist_wavelet   = np.array([0])
        self.alpha_hist_wavelet_f = np.array([0])
        log_spec        = np.log(spec)
        self.plots_info.load_alpha_info(self.freq, log_spec)
        # fig = plt.figure(figsize=(9,6))
        # # plt.subplot(1,1,1)
        # ax = fig.add_subplot(111)
        # ax.plot(log_spec, 'r')
        # plt.plot([self.alpha_index,  self.alpha_index], plt.gca().get_ylim(),'r--')
        # plt.title('alpha')
        # plt.draw()
        # plt.show()
        self.line_spectre.set_data(pos=np.concatenate([[np.arange(len(log_spec))], [log_spec]]).T)
        self.line_alpha_freq.set_data(pos=np.array([[self.alpha_index,np.min(log_spec)], [self.alpha_index,np.max(log_spec)]]), color="#ff0000")
        # self.beta_hist = np.array([0])
        # self.beta_hist_f = np.array([0])
        return self


class GUII(QtGui.QWidget):
    def __init__(self, time, pos, dt, apply_redis):
        QtGui.QWidget.__init__(self)
        #mainWindow = QtGui.QWidget()
        screenShape = QtGui.QDesktopWidget().screenGeometry()
        width  = screenShape.width()*9.9/10
        height = screenShape.height()*8./10
        self.resize(width, height)
        # Create the windows
        self.setWindowTitle("FullControl Alchemist")
        # Create the scatter
        self.canvas = RawPlot(time, pos, dt, apply_redis, size = (width,height))
        self.canvas.native.setParent(self)

def update_training(obj, a, last_index):
    global last_index_t
    global dt
    if obj.training_ongoing:
        if obj.canvas.seconds==0:
            obj.canvas.seconds=-1
            a.canvas.starting_sample = last_index
            # print(a.canvas.starting_sample)
        obj.update_training((last_index - last_index_t) * dt)
        if obj.training_done == True:
            a.canvas.end_sample = last_index
            # print(a.canvas.end_sample)
            a.training_ongoing = False
            obj.canvas.update_text('time of training is {0} s'.format((a.canvas.end_sample - a.canvas.starting_sample) * dt))
        
def update(event):
    global client
    global canals
    global timewindow
    global dt
    global last_index_t
    last_index   = client.getHeader().nSamples
    pos          = client.getData([(last_index-timewindow),(last_index-1)])[:,canals].T
    a.canvas.set_raw_data(time, pos)
    a.canvas.update_alpha()
    if a.canvas.training_window.training_ongoing:
        if a.canvas.training_window.canvas.seconds == 0:
            a.canvas.save_training_data(True, last_index)
            a.canvas.training_window.canvas.seconds = -1
        # print((last_index - last_index_t) * dt)
        a.canvas.training_window.update_training((last_index - last_index_t)*dt)
        if a.canvas.training_window.training_done == True:
            a.canvas.save_training_data(False, last_index)
            a.canvas.training_window.training_ongoing = False
            a.canvas.training_window.canvas.update_text('time of training is {0} s'.format((a.canvas.end_sample - a.canvas.starting_sample) * dt))
            a.canvas.training_window.__init__()
            a.canvas.train()
            #a.canvas.return_subjidx()
    update_training(a.canvas.beta_gui, a, last_index)
    if a.canvas.beta_gui.training_done:
        # print(a.canvas.starting_sample)
        # print(last_index)
        # print(a.canvas.end_sample)
        # print(canals)
        #pos          = client.getData([(last_index-timewindow),(last_index-1)])[:,canals].T
        print('time of training {0}'.format((a.canvas.end_sample - a.canvas.starting_sample) * dt))
        pos                         = client.getData([(a.canvas.starting_sample),(a.canvas.end_sample-1)])[:,canals].T
        a.canvas.quotient_spec_r, f = a.canvas.beta_gui.return_fft_comparison(pos, a.canvas.right_canal, dt)
        a.canvas.quotient_spec_l, f = a.canvas.beta_gui.return_fft_comparison(pos, a.canvas.left_canal, dt)
        a.canvas.plots_info.load_beta_r_info(f, a.canvas.quotient_spec_r)
        a.canvas.plots_info.load_beta_l_info(f, a.canvas.quotient_spec_l)
        pickle.dump(pos, open('data/subj{0}/left_right.pkl'.format(a.canvas.subj_idx), 'wb'), protocol=2)
        print('data of left/right saved')
        a.canvas.beta_gui.__init__()

    last_index_t = last_index

if __name__ == '__main__':
    global pos
    global time
    global rand
    global dt
    global generate_data
    global timewindow
    global last_index_t
    appQt         = QtGui.QApplication([])
    timewindow    = 1200
    generate_data = False
    dt            = 0.004
    Fs            = 1./dt
    time          = np.arange(timewindow)
    last_index_t  = 0
    apply_redis   = True

    if generate_data ==True:
        pos   = np.random.random((15,200))*2
        rand  = np.random.random((15))
        a     = GUII(time, pos, dt)
        timer = app.Timer(interval=dt, connect=update_generated, start=True) # otherwise interval="auto"
    else:
        import FieldTrip
        global client
        global canals
        # fieldtrip client
        client = FieldTrip.Client()
        #client.connect(hostname='10.30.33.151')
        client.connect(hostname='127.0.0.1') # client.connect(hostname='10.30.33.151') #IP ADDRESS TO GET DATA FROM !!!
        # canals of interest, last is reference
        canals = np.arange(0,4) #np.array([9, 16, 20, 13, 6, 5, 12, 19, 3, 10, 17, 23]) - 1
        #pos = np.random.random((15,200))*2
        # select data
        timelib.sleep(5.)
        print(client.getHeader().labels)
        last_index = client.getHeader().nSamples
        pos        = client.getData([(last_index-timewindow),(last_index-1)])[:,canals].T
        a          = GUII(time, pos, dt, apply_redis)
        timer      = app.Timer(interval=.1, connect=update, start=True) # otherwise interval="auto"
    a.show()
    appQt.exec_()

