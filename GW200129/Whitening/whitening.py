import readligo
import numpy as np
import matplotlib.pyplot as plt
from pycbc import types
from pycbc.psd import interpolate, welch
from pycbc.filter import highpass_fir, lowpass_fir

fig, ax = plt.subplots(3, 1, figsize = [11,6])
plt.subplots_adjust(left = 0.106, bottom = 0.112, right = 0.9, top = 0.9, wspace = 0.2, hspace = 0.962)

H1 = 'H-H1_GWOSC_4KHZ_R1-1264316101-32.hdf5'
L1 = 'L-L1_GWOSC_4KHZ_R1-1264316101-32.hdf5'
V1 = 'V-V1_GWOSC_4KHZ_R1-1264316101-32.hdf5'

i = 0

for data,dect in zip([H1, L1, V1],['H1','L1','V1']) :
	strain, time, channel_dict = readligo.loaddata(data)
	strain = types.TimeSeries(strain , delta_t = 1/4096)
	strain = highpass_fir(strain, 15, 8)
	psd = interpolate(welch(strain), delta_f = 1/strain.duration)
	whitened_strain = (strain.to_frequencyseries() / psd**0.5).to_timeseries()
	whitened_strain = highpass_fir(whitened_strain, 30, 8)
	whitened_strain = lowpass_fir(whitened_strain, 500, 8)
	ax[i].plot(time, whitened_strain, color = 'g')
	ax[i].set_xlabel('GPS time')
	ax[i].set_ylabel('whitened_strain_'+dect)
	ax[i].set_xlim([1264316116.4 - 0.1, 1264316116.4 + 0.1])
	ax[i].set_ylim([-180, 180])
	ax[i].grid(alpha = 0.5, color = 'r', ls = '--')
	i = i + 1

plt.show()
