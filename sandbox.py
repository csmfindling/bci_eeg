import numpy as np
import pickle


data = pickle.load(open('subjignacio3/left_right.pkl','rb'))


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

dt             = .004
channel        = 3 # right
nb_points      = len(data[channel])
idx            = int((nb_points/4)/1200)*1200
quartile       = int(nb_points/4)
spec_left , f  = fft_welsh(np.concatenate((data[channel][(quartile - idx):quartile], data[channel][(3 * quartile - idx):(3 * quartile)])), dt)
spec_right, f  = fft_welsh(np.concatenate((data[channel][(2 * quartile - idx):(2 * quartile)], data[channel][-idx:])), dt)
freq_interest  = f[(f>6)*(f<40)]
spec_right_int = spec_right[(f>6)*(f<40)]
spec_left_int  = spec_left[(f>6)*(f<40)]

dq = spec_right - spec_left
sq = spec_right + spec_left
dq = dq[(f>6)*(f<40)]
sq = sq[(f>6)*(f<40)]

plt.plot(spec_right_int)
plt.plot(spec_left_int)
plt.plot(dq)
plt.plot(sq)

plt.plot(freq_interest, -dq/sq)

dt             = .004
channel        = 4 # left
nb_points      = len(data[channel])
idx            = int((nb_points/4)/1200)*1200
quartile       = int(nb_points/4)
spec_left , f  = fft_welsh(np.concatenate((data[channel][(quartile - idx):quartile], data[channel][(3 * quartile - idx):(3 * quartile)])), dt)
spec_right, f  = fft_welsh(np.concatenate((data[channel][(2 * quartile - idx):(2 * quartile)], data[channel][-idx:])), dt)
freq_interest  = f[(f>6)*(f<40)]
spec_right_int = spec_right[(f>6)*(f<40)]
spec_left_int  = spec_left[(f>6)*(f<40)]

dq = spec_right - spec_left
sq = spec_right + spec_left
dq = dq[(f>6)*(f<40)]
sq = sq[(f>6)*(f<40)]
plt.plot(spec_right_int)
plt.plot(spec_left_int)

plt.plot(freq_interest, dq/sq)

