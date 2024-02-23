import readligo
import numpy as np
import matplotlib.pyplot as plt
from pycbc import types
from pycbc.psd import interpolate, welch
from pycbc.filter import highpass_fir, lowpass_fir

fig, ax = plt.subplots(3, 1, figsize = [11,6])
plt.subplots_adjust(left = 0.131, bottom = 0.112, right = 0.87, top = 0.9, wspace = 0.2, hspace = 0.962)

H1 = 'H-H1_GWOSC_4KHZ_R1-1264316101-32.hdf5'
L1 = 'L-L1_GWOSC_4KHZ_R1-1264316101-32.hdf5'
V1 = 'V-V1_GWOSC_4KHZ_R1-1264316101-32.hdf5'

i = 0

for data,dect in zip([H1, L1, V1],['H1','L1','V1']) :
	strain, time, channel_dict = readligo.loaddata(data)
	strain = types.TimeSeries(strain , delta_t = 1/4096)
	strain = highpass_fir(strain, 15, 8)
	strain = strain.crop(2,2)
	psd = interpolate(welch(strain), delta_f = 1/strain.duration)
	whitened_strain = (strain.to_frequencyseries() / psd**0.5).to_timeseries()
	whitened_strain = whitened_strain.time_slice(14, 16)
	times, freq ,power = whitened_strain.qtransform(0.001,logfsteps=200,frange=(30,500),qrange=(10,10))
	f = ax[i].pcolormesh(times, freq, power, vmax = 25, cmap = 'inferno')
	ax[i].set_yscale('log')
	ax[i].set_xlabel('Sample time (second)')
	ax[i].set_ylabel('Frequency_'+dect)
	ax[i].grid(alpha = 0.5, color = 'white')
	i = i + 1

cbar_ax = fig.add_axes([0.91, 0.11, 0.017, 0.79])
cbar=fig.colorbar(f, cax=cbar_ax)
plt.show()
