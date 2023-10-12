import readligo
import matplotlib.pyplot as plt

fig, ax = plt.subplots(3, 1, figsize = [11,6])
plt.subplots_adjust(left=0.106, bottom=0.112, right=0.9, top=0.9, wspace=0.2, hspace=0.962)

H1 = 'H-H1_GWOSC_4KHZ_R1-1264316101-32.hdf5'
L1 = 'L-L1_GWOSC_4KHZ_R1-1264316101-32.hdf5'
V1 = 'V-V1_GWOSC_4KHZ_R1-1264316101-32.hdf5'

i = 0

for data,dect in zip([H1, L1, V1],['H1','L1','V1']) :
	strain, time, channel_dict = readligo.loaddata(data)
	ax[i].plot(time, strain)
	ax[i].set_xlabel('GPS time')
	ax[i].set_ylabel('Strain_'+dect)
	ax[i].grid(alpha = 0.2, color = 'brown')
	i = i + 1

plt.show()
