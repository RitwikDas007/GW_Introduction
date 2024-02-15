import readligo
import numpy as np
import matplotlib.pyplot as plt
from pycbc import types, waveform
from pycbc.psd import interpolate, welch
from pycbc.filter import highpass_fir, lowpass_fir, matched_filter
from pycbc.filter import sigma

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
SNR = matched_filter(tem_p, strain, psd = psd, low_frequency_cutoff = 20)
SNR = SNR.time_slice(8, 20)
time = GPS_time.time_slice(8, 20)
pos = np.argmax(SNR)
peak = time[pos]
dt = peak - GPS_time[0]

aligned = tem_p.cyclic_time_shift(dt)
aligned /= sigma(aligned, psd=psd, low_frequency_cutoff=30)

aligned = (aligned * SNR[pos]).to_timeseries()
aligned.start_time = GPS_time[0]
strain.start_time = GPS_time[0]

white_data = (strain.to_frequencyseries() / psd**0.5).to_timeseries()
white_template = (aligned.to_frequencyseries() / psd**0.5).to_timeseries()
white_data = white_data.highpass_fir(30,8).lowpass_fir(500, 8)
white_template = white_template.highpass_fir(30, 8).lowpass_fir(500, 8)

plt.figure(figsize=[15, 4])
plt.plot(white_data.sample_times, white_data, color = 'brown', lw=0.8, label="Data")
plt.plot(white_template.sample_times,white_template, lw=0.9, label="Template")
plt.xlim([1264316116.4 - 0.1, 1264316116.4 + 0.08])
plt.ylim([-180, 180])
plt.xlabel('GPS time')
plt.ylabel('Whitened _strain_L1')
plt.legend()
plt.show()
