from astropy.io import fits
from numpy import *
from scipy import fftpack, optimize
from matplotlib import pyplot as plt
from multiprocessing import Pool
import seaborn as sns
import os

class spec:
        def __init__(self, fname, ra, dec):
                self.fname = fname
		self.ra = ra
		self.dec = dec
                self.read_spec()
                #self.remove_con()

        def read_spec(self):
                data = fits.open(self.fname)[0].data
                self.flux = data[0]
                self.wavelength = data[2]

        def remove_con(self):
                fprime = fftpack.rfft(self.flux.astype('f4'))
                fprime[30:] = 0
                self.con = fftpack.irfft(fprime)
                self.flux_nocon = self.flux - self.con

        def plot(self):
		plt.figure(figsize=(15, 8))
		plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
		plt.rc('ytick', labelsize=12) 
		font = {'size':'14'}
		plt.xticks(arange(3500, 9200, 500))
		plt.xlabel('wavelength (Angstrom)', **font)
		plt.ylabel('flux', **font)
                plt.plot(self.wavelength, self.flux)
		#plt.show()
		plt.savefig('{0}_{1}_{2}.png'.format(self.fname.split('/')[-1], self.ra, self.dec))
	def output(self):
		out = ''
		for n, i in enumerate(self.wavelength):
			out += '{0} {1}\n'.format(i, self.flux[n])
		outfile = open('{0}_{1}_{2}.txt'.format(self.fname.split('/')[-1], self.ra, self.dec ), 'w')
		outfile.write(out)
		outfile.close()
		
			


def main():
	pn_list = open('pn0515.txt').readlines()
	for i in pn_list:
		i=i.split(', ')
		print i
		try :
			plot = spec(i[0], i[1], i[2])
			plot.plot()
			plot.output()
		except IOError:
			pass

if __name__ == "__main__":
        main()
