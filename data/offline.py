import numpy as np
import pickle
import os
import sys
sys.path.append('../')
import functions

data         = pickle.load(open('subjch0/left_right.pkl', 'rb'))
left_elect   = data[2]
middle_elect = data[3]
right_elect  = data[4]

T        = int(len(left_elect))
times    = int(T/4)
dt       = .004
interval = int((int(T/4)-int(T/4)%1200))

mouv1_leftelect, f   = functions.fft_welsh(left_elect[int(T/4 - interval):int(T/4)], dt)
mouv1_rightelect, f  = functions.fft_welsh(right_elect[int(T/4 - interval):int(T/4)], dt)
mouv1_middleelect, f = functions.fft_welsh(middle_elect[int(T/4 - interval):int(T/4)], dt)
mouv1                = (mouv1_leftelect + mouv1_rightelect + mouv1_middleelect)/3.

nomouv1_leftelect, f   = functions.fft_welsh(left_elect[int(T/2 - interval):int(T/2)], dt)
nomouv1_rightelect, f  = functions.fft_welsh(right_elect[int(T/2 - interval):int(T/2)], dt)
nomouv1_middleelect, f = functions.fft_welsh(middle_elect[int(T/2 - interval):int(T/2)], dt)
nomouv1                = (nomouv1_leftelect + nomouv1_rightelect + nomouv1_middleelect)/3.

f_interest = (f > 8) * (f < 30)

plt.plot(f[f_interest], mouv1_leftelect[f_interest])
plt.plot(f[f_interest], nomouv1_leftelect[f_interest])
plt.plot(f[f_interest], mouv1_rightelect[f_interest])
plt.plot(f[f_interest], nomouv1_rightelect[f_interest])

mouv2_leftelect, f   = functions.fft_welsh(left_elect[int(3*T/4 - interval):int(3*T/4)], dt)
mouv2_rightelect, f  = functions.fft_welsh(right_elect[int(3*T/4 - interval):int(3*T/4)], dt)
mouv2_middleelect, f = functions.fft_welsh(middle_elect[int(3*T/4 - interval):int(3*T/4)], dt)
mouv2                = (mouv2_leftelect + mouv2_rightelect + mouv2_middleelect)/3.

nomouv2_leftelect, f   = functions.fft_welsh(left_elect[int(T - interval):int(T)], dt)
nomouv2_rightelect, f  = functions.fft_welsh(right_elect[int(T - interval):int(T)], dt)
nomouv2_middleelect, f = functions.fft_welsh(middle_elect[int(T - interval):int(T)], dt)
nomouv2                = (nomouv2_leftelect + nomouv2_rightelect + nomouv2_middleelect)/3.

f_interest = (f > 8) * (f < 30)

plt.plot(f[f_interest], mouv2_leftelect[f_interest])
plt.plot(f[f_interest], nomouv2_leftelect[f_interest])
plt.plot(f[f_interest], mouv2_rightelect[f_interest])
plt.plot(f[f_interest], nomouv2_rightelect[f_interest])