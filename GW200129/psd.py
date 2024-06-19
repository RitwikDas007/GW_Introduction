import readligo
import numpy as np
import matplotlib.pyplot as plt
from pycbc import types
from pycbc.psd import interpolate, welch
from pycbc.filter import highpass_fir

H1 = 'H-H1_GWOSC_4KHZ_R1-1264316101-32.hdf5'
L1 = 'L-L1_GWOSC_4KHZ_R1-1264316101-32.hdf5'
V1 = 'V-V1_GWOSC_4KHZ_R1-1264316101-32.hdf5'

for data,dect in zip([H1, L1, V1],['H1','L1','V1']) :
	strain, time, channel_dict = readligo.loaddata(data)
	strain = types.TimeSeries(strain , delta_t = 1/4096)
	strain = highpass_fir(strain, 15, 8)
	psd = interpolate(welch(strain), delta_f = 1/32)
	plt.loglog(psd.sample_frequencies, psd, label = dect)

plt.xlabel('Frequency (Hz)')
plt.ylabel('$Strain^{2}$ / Hz')
plt.grid(alpha = 0.4, color = 'grey', ls='--')
plt.xlim(20,1000)
plt.legend()
plt.show()
