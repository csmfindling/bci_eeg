import numpy as np
from scipy.signal import convolve

def get_dilated_morlet(M, w=5.0, s=1., dt=0.004):
	x       = np.linspace(-M/2,M/2,M+1) * dt #np.linspace(-2*np.pi, 2*np.pi, M) * s
	output  = np.exp(1j * w * x * s)
	output -= np.exp(-0.5 * (w**2))
	output *= np.exp(-0.5 * ((x*s)**2)) * np.pi**(-0.25)
	output *= np.power(1 + np.exp(-w**2) - 2 * np.exp(-3./4 * w**2) , -1./2)
	output *= s
	return [x, output]

def get_dilated_morlet_fourier(M, w=5.0, s=1., unif=False):
	#w0      = w*s
	if unif==True:
		x       = np.linspace(0, .5 * np.pi * 2, M)/s
	else:
		x       = np.linspace(0, 50 * np.pi * 2, M)/s
	output  = np.exp(-1./2*(w - x)**2)
	output -= np.exp(-0.5 * (w**2)) * np.exp(-0.5 * (x**2))
	output *= np.power(1 + np.exp(-w**2) - 2 * np.exp(-3./4 * w**2) , -1./2) * np.pi**(-0.25)
	if unif==True:
		return [np.linspace(0, .5 * np.pi * 2, M), output]
	else:
		return [np.linspace(0, 50 * np.pi * 2, M), output]

class wavelet_transform:

	# filter bank
	def __init__(self):
		self.M           = 50
		self.nb_filter   = 52
		self.s_list_f    = 2**(np.arange(self.nb_filter)/12.)
		self.w_list_f    = np.ones(self.nb_filter)*2.*2*np.pi
		self.dt          = 0.004
		self.wavelets    = np.zeros([self.nb_filter, 2, self.M])
		self.frequencies = self.w_list_f*self.s_list_f/(2*np.pi)
		self.freq_ran    = np.array([1,4,7,13,18,25,30,40])
		for i in range(len(self.w_list_f)):
			x, morlet_d = get_dilated_morlet(M=self.M-1, w=self.w_list_f[i],s=self.s_list_f[i], dt=self.dt)
			self.wavelets[i,0] = np.real(morlet_d)
			self.wavelets[i,1] = np.imag(morlet_d)

	def transform(self, datapoint):
		nb_chan, l_data = datapoint.shape
		nb_valid        = l_data - self.M + 1
		convolutions    = np.zeros([nb_chan, self.nb_filter, nb_valid, 2])
		for chan_idx in range(nb_chan):
			for filter_idx in range(self.nb_filter):
				convolutions[chan_idx, filter_idx, :, 0] = np.convolve(datapoint[chan_idx], self.wavelets[filter_idx,0], mode='valid')
				convolutions[chan_idx, filter_idx, :, 1] = np.convolve(datapoint[chan_idx], self.wavelets[filter_idx,1], mode='valid')
		modulus = np.sqrt(convolutions[:,:,:,0]**2 + convolutions[:,:,:,1]**2)
		nb_bins  = len(self.freq_ran) - 1
		average  = np.zeros([nb_bins, nb_valid])
		# for idx_chan in range(nb_chan):
		for idx_bin in range(nb_bins):
			freq_idx = (self.frequencies >= self.freq_ran[idx_bin]) * (self.frequencies < self.freq_ran[idx_bin + 1])
			for idx_valid in range(nb_valid):
				average[idx_bin, idx_valid] = np.mean(modulus[0, freq_idx, idx_valid] - modulus[1, freq_idx, idx_valid])
		average2    = np.mean(average, axis=1)
		standardize = (average2 - np.mean(average2))/np.std(average2)
		# standardize = np.zeros([nb_chan, nb_bins])
		# for idx_bin in range(nb_bins):
		# 	standardize[:, idx_bin] = (average2[:, idx_bin] - np.mean(average2[:, idx_bin]))/(np.std(average2[:, idx_bin]))
		assert(not np.isnan(np.sum(standardize)))
		return standardize






