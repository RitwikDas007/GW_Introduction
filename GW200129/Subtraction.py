import readligo
import numpy as np
import matplotlib.pyplot as plt
from pycbc import types, waveform
from pycbc.psd import interpolate, welch
from pycbc.filter import highpass_fir, lowpass_fir, matched_filter
from pycbc.filter import sigma

'''
fig = plt.figure(figsize = [12,6])
ax1 = plt.axes([0.12,0.6,0.8,0.3])
ax2 = plt.axes([0.12,0.2,0.8,0.3])'''

fig, ax = plt.subplots(2, 1, figsize = [11,6])
plt.subplots_adjust(left = 0.131, bottom = 0.112, right = 0.87, top = 0.9, wspace = 0.2, hspace = 0.962)

L1 = 'L-L1_GWOSC_4KHZ_R1-1264316101-32.hdf5'

strain, time, channel_dict = readligo.loaddata(L1)
GPS_time = types.TimeSeries(time , delta_t = 1/4096)
strain = types.TimeSeries(strain , delta_t = 1/4096)
strain = highpass_fir(strain, 15, 8)
starin = strain.crop(2,2)
psd = interpolate(welch(strain), 1/strain.duration)
strain = strain.to_frequencyseries()
tem_p, tem_c = waveform.get_fd_waveform(approximant = 'IMRPhenomD' ,
					mass1 = 36 ,
					mass2 = 34 ,
					delta_f = 1/strain.duration ,
					f_lower = 30)
tem_p.resize(len(strain))
SNR = matched_filter(tem_p, strain, psd = psd, low_frequency_cutoff = 30)
SNR = SNR.time_slice(8, 20)
time = GPS_time.time_slice(8, 20)
SNR = abs(SNR)
pos = np.argmax(SNR)
peak = time[pos]

dt = peak - GPS_time[0]
aligned = tem_p.cyclic_time_shift(dt)
aligned /= sigma(aligned, psd=psd, low_frequency_cutoff=30)


aligned = (aligned * max(SNR)).to_timeseries()
aligned.start_time = GPS_time[0]

white_data = (strain.to_frequencyseries() / psd**0.5).to_timeseries()
white_template = (aligned.to_frequencyseries() / psd**0.5).to_timeseries()
white_data = white_data.highpass_fir(30,8).lowpass_fir(500, 8)
white_template = white_template.highpass_fir(30, 8).lowpass_fir(500, 8)

white_data.start_time = white_template.start_time
white_subtracted = white_data - white_template
white_subtracted = white_subtracted.crop(14,16)
white_data = white_data.crop(14,16)

times, freq ,power = white_subtracted.qtransform(0.001,logfsteps=200,frange=(30,500),qrange=(10,10))

i = 0

for data, title in zip([white_data, white_subtracted], ['L1 Data', 'Signal Subtracted from L1 Data']):
	times, freq ,power = data.qtransform(0.001,logfsteps=200,frange=(30,500),qrange=(10,10))
	f = ax[i].pcolormesh(times, freq, power, vmax=25, cmap='inferno')
	ax[i].set_yscale('log')
	ax[i].set_xlabel('GPS Time')
	ax[i].set_ylabel('Frequency')
	ax[i].grid(alpha = 0.5, color = 'white')
	ax[i].set_title(title)
	i = i + 1

cbar_ax = fig.add_axes([0.91, 0.11, 0.017, 0.79])
cbar=fig.colorbar(f, cax=cbar_ax)
plt.show()


