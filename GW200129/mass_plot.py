import readligo
import numpy as np
import matplotlib.pyplot as plt
from pycbc import types, waveform
from pycbc.psd import interpolate, welch
from pycbc.filter import highpass_fir, lowpass_fir, matched_filter

fig,ax = plt.subplots(1, 3, figsize=[11,5])
plt.subplots_adjust(left =0.065, bottom = 0.226, right = 0.97, top = 0.75, wspace = 0.455, hspace = 0.2)

H1 = 'H-H1_GWOSC_4KHZ_R1-1264316101-32.hdf5'
L1 = 'L-L1_GWOSC_4KHZ_R1-1264316101-32.hdf5'
V1 = 'V-V1_GWOSC_4KHZ_R1-1264316101-32.hdf5'

mass = np.linspace(15, 60, 40)
n = len(mass)
snr = np.zeros((n,n))

z = 0

for data,dect in zip([H1, L1, V1],['H1','L1','V1']) :
	strain, time, channel_dict = readligo.loaddata(data)
	time = types.TimeSeries(time , delta_t = 1/4096)
	strain = types.TimeSeries(strain , delta_t = 1/4096)
	strain = highpass_fir(strain, 15, 8)
	psd = interpolate(welch(strain), 1/strain.duration)
	strain = highpass_fir(strain, 36, 8)
	strain = lowpass_fir(strain, 500, 8)
	strain = strain.to_frequencyseries()
	for i in range(n):
		for j in range(i+1):
			mass1 = mass[i]
			mass2 = mass[j]
			sp,sc = waveform.get_fd_waveform(approximant = 'IMRPhenomD' ,
							mass1 = mass1 ,
							mass2 = mass2 ,
							delta_f = 1/strain.duration ,
							f_lower = 20)
			sp.resize(len(strain))
			SNR = matched_filter(sp, strain, psd=psd, low_frequency_cutoff=20)
			SNR = SNR.time_slice(14,18)
			SNR = abs(SNR)
			snr[i,j] = max(SNR)		
	f = ax[z].contourf(mass, mass, snr, cmap = 'BrBG')
	ax[z].set_title(dect)
	ax[z].set_xlabel('Mass_2')
	ax[z].set_ylabel('Mass_1')
	ax[z].set_xticks([20,30,40,50,60])
	ax[z].set_yticks([20,30,40,50,60])
	fig.colorbar(f)
	z = z+1	

plt.show()
