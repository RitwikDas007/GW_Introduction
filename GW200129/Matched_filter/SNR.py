import readligo
import numpy as np
import matplotlib.pyplot as plt
from pycbc import types, waveform
from pycbc.psd import interpolate, welch
from pycbc.filter import highpass_fir, lowpass_fir, matched_filter

H1 = 'H-H1_GWOSC_4KHZ_R1-1264316101-32.hdf5'
L1 = 'L-L1_GWOSC_4KHZ_R1-1264316101-32.hdf5'
V1 = 'V-V1_GWOSC_4KHZ_R1-1264316101-32.hdf5'

i = 0

for data,dect in zip([H1, L1, V1],['H1','L1','V1']) :
	strain, time, channel_dict = readligo.loaddata(data)
	time = types.TimeSeries(time , delta_t = 1/4096)
	strain = types.TimeSeries(strain , delta_t = 1/4096)
	strain = highpass_fir(strain, 15, 8)
	psd = interpolate(welch(strain), 1/strain.duration)
	strain = highpass_fir(strain, 30, 8)
	strain = lowpass_fir(strain, 500, 8)
	strain = strain.to_frequencyseries()
	tem_p, tem_c = waveform.get_fd_waveform(approximant = 'IMRPhenomD' ,
					mass1 = 34.5 ,
					mass2 = 28.9 ,
					delta_f = 1/strain.duration ,
					f_lower = 20)
	tem_p.resize(len(strain))
	SNR = matched_filter(tem_p, strain, psd = psd, low_frequency_cutoff = 20)
	SNR = SNR.time_slice(8, 20)
	time = time.time_slice(8, 20)
	plt.plot(time, abs(SNR), label = dect)
	plt.xlabel('GPS time')
	plt.ylabel('Signal-to-noise ratio')
	i = i + 1

plt.grid(alpha = 0.3, color = 'grey')
plt.legend()
plt.show()
