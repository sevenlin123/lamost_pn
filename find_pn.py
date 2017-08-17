from astropy.io import fits
from numpy import *
from scipy import fftpack, optimize
from matplotlib import pyplot as plt
from multiprocessing import Pool
import os

class spec:
	def __init__(self, fname):
		self.fname = fname
		self.read_spec()
		self.remove_con()
		
	def read_spec(self):
		spec = fits.open(self.fname)[0]
		header = spec.header
		data = spec.data 
		self.flux = data[0]
		self.wavelength = data[2]
		self.ra = header['ra']
		self.dec = header['dec'] 
		
	def remove_con(self):
		fprime = fftpack.rfft(self.flux.astype('f4'))
		fprime[30:] = 0
		self.con = fftpack.irfft(fprime)
		self.flux_nocon = self.flux - self.con

	def plot(self):
		#plt.plot(self.wavelength, self.flux)
		plt.plot(self.wavelength, self.con)
		plt.plot()
		#plt.plot(self.wavelength, self.flux_nocon)
		#plt.plot()
	
class emission:
	def __init__(self, wl, flux, which):
		self.window = {'Ha_N2': logical_and(wl < 6600, wl >6500), 'o3': logical_and(wl < 5050, wl >4930),
				 'S2': logical_and(wl < 6750, wl >6700), 'Hb': logical_and(wl < 4870, wl >4850)}
		#self.oiii_window = logical_and(wl < 5050, wl >4930)
		
		self.data = flux[self.window[which]]
		self.wl = wl[self.window[which]]
		if which == 'o3':
			self.fit_o3()
		elif which == 'Ha_N2':
			self.fit_Ha_N2()
		elif which == 'S2':
			self.fit_S2()
		elif which == 'Hb':
			self.fit_Hb()
		#self.plot()
	def model_1gaussian(self, x0, flux0, bg0, bg1, sigma0):
        #sigma0 = 1.1
                return lambda x: flux0 / (2*pi*sigma0**2)**0.5 * exp(-(x-x0)**2/(2*sigma0**2))+\
                                                 bg0 + bg1*x

	def model_2gaussian(self, x0, x1, flux0, flux1, bg0, bg1, sigma0, sigma1):
        #sigma0 = 1
        #sigma1 = 1
		return lambda x: flux0 / (2*pi*sigma0**2)**0.5 * exp(-(x-x0)**2/(2*sigma0**2))+\
                                                 flux1 / (2*pi*sigma1**2)**0.5 * exp(-(x-x1)**2/(2*sigma1**2)) +\
                                                 bg0 + bg1*x

	def model_3gaussian(self, x0, x1, x2, flux0, flux1, flux2, bg0, bg1, sigma0, sigma1, sigma2, flux3, sigma3):
        #sigma0 = 1
        #sigma1 = 1.5
        #sigma2 = 1
		return lambda x: flux0 / (2*pi*sigma0**2)**0.5 * exp(-(x-x0)**2/(2*sigma0**2))+\
						 flux1 / (2*pi*sigma1**2)**0.5 * exp(-(x-x1)**2/(2*sigma1**2)) +\
						 flux2 / (2*pi*sigma2**2)**0.5 * exp(-(x-x2)**2/(2*sigma2**2)) +\
						 bg0 + bg1*x +\
                         flux3 / (2*pi*sigma3**2)**0.5 * exp(-(x-x0)**2/(2*sigma3**2))

	def fit_o3(self):
		params = [28, 68, 10000, 10000, 0, 0, 1, 1]
		errorfunction = lambda p: ravel(self.model_2gaussian(*p)(indices(self.data.shape)) - self.data)
		p, cov, infodict, errmsg, success = optimize.leastsq(errorfunction, params, full_output=1, col_deriv=1)
		self.fit_result = p
		self.fit_model = self.model_2gaussian(*p)
		try:
			s_sq = (errorfunction(self.fit_result)**2).sum()/(2*len(self.data))**2
			cov = cov*s_sq
			fit_err = diag(cov)**0.5
            		self.peak0 = self.wl[int(p[0])]
            		self.peak1 = self.wl[int(p[1])]
            		self.flux0 = p[2]
			self.flux0_err = fit_err[2]
            		self.flux1 = p[3]
			self.flux1_err = fit_err[3]
          		self.flux_ratio = p[2]/p[3]
            		self.peak_flux0 = self.data[int(p[0])]
            		self.peak_flux1 = self.data[int(p[1])]
            		self.residual = self.data-self.fit_model(arange(0, self.data.size))
            		#print self.peak0, self.peak1, self.flux0, self.flux1, self.flux_ratio
            		self.O3 = abs(self.peak0 - 4959) < 5 and abs(self.peak1 - 5007) < 5 and (self.peak_flux0) > 3*self.residual.std() and (self.peak_flux1) > 3*self.residual.std()
            		#print self.O3
        	except IndexError:
            		self.O3 = False
			self.flux0 = 'Nan'
			self.flux1 = 'Nan'
			self.flux0_err = 'Nan'
			self.flux1_err = 'Nan'
                except TypeError:
                        self.O3 = False
                        self.flux0 = 'Nan'
                        self.flux1 = 'Nan'
                        self.flux0_err = 'Nan'
                        self.flux1_err = 'Nan'
 
        def fit_Ha_N2(self):
                params_ha = [42, 20000, 0, 0, 1.5]
                errorfunction = lambda p: ravel(self.model_1gaussian(*p)(indices(self.data.shape)) - self.data)
                p, cov, infodict, errmsg, success = optimize.leastsq(errorfunction, params_ha, full_output=1, col_deriv=1)
		self.fit_result = p
                self.fit_model_ha = self.model_1gaussian(*p)
                try:
			s_sq = (errorfunction(self.fit_result)**2).sum()/(2*len(self.data))**2
			cov = cov*s_sq
			fit_err = diag(cov)**0.5
			self.peak0 = self.wl[int(p[0])]
			self.flux0 = p[1]
			self.Ha_flux_err = fit_err[1]
			#self.flux1 = p[2]
			#self.flux1_err = fit_err[2]
			self.residual = self.data-self.fit_model_ha(arange(0, self.data.size))
			params_n2 = [33, 56, 1000, 1000, 0, 0, 1, 1]                
    			errorfunction_n2 = lambda p: ravel(self.model_2gaussian(*p)(indices(self.data.shape)) - self.residual)
	                p_n2, cov_n2, infodict_n2, errmsg_n2, success_n2 = optimize.leastsq(errorfunction_n2, params_n2, full_output=1, col_deriv=1)
        	        self.fit_result_n2 = p_n2 
			s_sq_n2 = (errorfunction_n2(self.fit_result_n2)**2).sum()/(2*len(self.data))**2
			cov_n2 = cov_n2*s_sq_n2
			fit_err_n2 = diag(cov_n2)**0.5
			self.fit_model_n2 = self.model_2gaussian(*p_n2)
			self.peakn2_0 = self.wl[int(p_n2[0])]
			self.peakn2_1 = self.wl[int(p_n2[1])]
			self.fluxn2_0 = p_n2[2]
			self.fluxn2_0_err = fit_err_n2[2]
			self.fluxn2_1 = p_n2[3]
			self.fluxn2_1_err = fit_err_n2[3]
			self.peak_fluxn2_0 = self.residual[int(p_n2[0])]
			self.peak_fluxn2_1 = self.residual[int(p_n2[1])]
			self.residual_n2 = self.residual-self.fit_model_n2(arange(0, self.data.size))
			self.Ha_flux = self.flux0
			self.Ha = abs(self.peak0 - 6563) < 5 and self.Ha_flux > 100
			self.N2 = abs(self.peakn2_0 - 6548) < 5 and abs(self.peakn2_1 - 6584) and self.fluxn2_0 > 50 and self.fluxn2_1 > 50
                    	#print self.fluxn2_0, self.fluxn2_1, self.flux0, self.Ha, self.N2
			self.fit_model = 0#elf.fit_model_ha + self.fit_model_n2
                except IndexError:
			self.Ha = False
			self.N2 = False
			self.Ha_flux = 'Nan'
			self.fluxn2_0 = 'Nan'
			self.fluxn2_1 = 'Nan'
			self.Ha_flux_err = 'Nan'
			self.fluxn2_0_err = 'Nan'
			self.fluxn2_1_err = 'Nan'
		except TypeError:
                        self.Ha = False
                        self.N2 = False
                        self.Ha_flux = 'Nan'
                        self.fluxn2_0 = 'Nan'
                        self.fluxn2_1 = 'Nan'
                        self.Ha_flux_err = 'Nan'
                        self.fluxn2_0_err = 'Nan'
                        self.fluxn2_1_err = 'Nan'

        def fit_S2(self):
                params = [12, 21, 10000, 10000, 0, 0, 1, 1]
                errorfunction = lambda p: ravel(self.model_2gaussian(*p)(indices(self.data.shape)) - self.data)
                p, cov, infodict, errmsg, success = optimize.leastsq(errorfunction, params, full_output=1, col_deriv=1)
		self.fit_result = p
                self.fit_model = self.model_2gaussian(*p)
                try:
			s_sq = (errorfunction(self.fit_result)**2).sum()/(2*len(self.data))**2
                        cov = cov*s_sq
                        fit_err = diag(cov)**0.5
			self.peak0 = self.wl[int(p[0])]
			self.peak1 = self.wl[int(p[1])]
			self.flux0 = p[2]
			self.flux0_err = fit_err[2]
			self.flux1 = p[3]
			self.flux1_err = fit_err[3]
			self.flux_ratio = p[2]/p[3]
			self.peak_flux0 = self.data[int(p[0])]
			self.peak_flux1 = self.data[int(p[1])]
			self.residual = self.data-self.fit_model(arange(0, self.data.size))
			self.S2 = abs(self.peak0 - 6717) < 5 and abs(self.peak1 - 6731) < 5 and (self.peak_flux0) > 3*self.residual.std() and (self.peak_flux1) > 3*self.residual.std()
                    #print self.S2
                except IndexError:
			self.S2 = False
			self.flux0 = 'Nan'
			self.flux1 = 'Nan'
			self.flux0_err = 'Nan'
			self.flux1_err = 'Nan'
                except TypeError:
                        self.S2 = False
                        self.flux0 = 'Nan'
                        self.flux1 = 'Nan'
                        self.flux0_err = 'Nan'
                        self.flux1_err = 'Nan'

        def fit_Hb(self):
                params = [11,  10000, 0, 0, 2]
		errorfunction = lambda p: ravel(self.model_1gaussian(*p)(indices(self.data.shape)) - self.data)
                p, cov, infodict, errmsg, success = optimize.leastsq(errorfunction, params, full_output=1, col_deriv=1)
		self.fit_result = p
                self.fit_model = self.model_1gaussian(*p)
                try:
			s_sq = (errorfunction(self.fit_result)**2).sum()/(2*len(self.data))**2
			cov = cov*s_sq
			fit_err = diag(cov)**0.5
			self.peak0 = self.wl[int(p[0])]
			self.flux0 = p[1]
			#self.flux1 = p[2]
			self.residual = self.data-self.fit_model(arange(0, self.data.size))
			self.Hb_flux = self.flux0
			self.Hb_flux_err = fit_err[1]
			self.Hb = abs(self.peak0 - 4861) < 5 and (self.Hb_flux) > 3*self.residual.std()
                    #print self.Hb_flux, self.Hb
                except IndexError:
			self.Hb = False
			self.Hb_flux = 'Nan'
			self.Hb_flux_err = 'Nan'
		except TypeError:
                        self.Hb = False
                        self.Hb_flux = 'Nan'
                        self.Hb_flux_err = 'Nan'

	def plot(self):
		plt.plot(self.data)
		if self.fit_model == 0:
			plt.plot(self.fit_model_n2(arange(0, self.data.size))+self.fit_model_ha(arange(0, self.data.size)))
		else:
			plt.plot(self.fit_model(arange(0, self.data.size)))
		plt.show()
		
def find_PN(i):
	try:
		spec0 = spec(i)
		o3 = emission(spec0.wavelength, spec0.flux_nocon, 'o3')
		Ha_N2 = emission(spec0.wavelength, spec0.flux_nocon, 'Ha_N2')
		S2 = emission(spec0.wavelength, spec0.flux_nocon, 'S2')
		Hb = emission(spec0.wavelength, spec0.flux_nocon, 'Hb')
		PN_cand = o3.O3 and Ha_N2.Ha and Ha_N2.N2 and S2.S2 and Hb.Hb
		return "{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9} {10}, {11}, {12}, {13}, {14}, {15}, {16}, {17}, {18}, {19}, {20}, {21}, {22}, {23}, {24}".format(
			i, spec0.ra, spec0.dec, Ha_N2.Ha_flux, Ha_N2.fluxn2_0, Ha_N2.fluxn2_1, Hb.Hb_flux, S2.flux0, S2.flux1, 
			o3.flux0, o3.flux1, Ha_N2.Ha_flux_err, Ha_N2.fluxn2_0_err, Ha_N2.fluxn2_1_err, Hb.Hb_flux_err, S2.flux0_err, 
			S2.flux1_err, o3.flux0_err, o3.flux1_err, Ha_N2.Ha, Ha_N2.N2, Hb.Hb, S2.S2, o3.O3, PN_cand)
	except IOError:
		pass

def main():
	fits_list = [i.strip() for i in open('fits.list').readlines()]#filter(lambda x: x.endswith('fits.gz'), os.listdir('.'))
	print "#file_name RA DEC  Ha_flux N2_flux1 N2_flux2 Hb_flux S2_flux1 S2_flux2 O3_flux1 O3_flux2 Ha_flux_err N2_flux1_err N2_flux2_err Hb_flux_err S2_flux1_err S2_flux2_err O3_flux1_err O3_flux2_err Ha_emission N2_emission Hb_emission S2_emission O3_emission PN?"
	p = Pool(24)
	#fits_list = ['./M31_014139N300249_F2/spec-55913-M31_014139N300249_F2_sp14-203.fits.gz']
	result = p.map(find_PN, fits_list)
	for i in result:
		print i

	#for i in fits_list:
	#	spec0 = spec(i) #spec('spec-55976-GAC_099N04_V1_sp03-053.fits.gz')
	#	spec0.plot()
	#	o3 = emission(spec0.wavelength, spec0.flux_nocon, 'o3')
	#	Ha_N2 = emission(spec0.wavelength, spec0.flux_nocon, 'Ha_N2')
	#	S2 = emission(spec0.wavelength, spec0.flux_nocon, 'S2')
	#	Hb = emission(spec0.wavelength, spec0.flux_nocon, 'Hb')
	#	PN_cand = o3.O3 and Ha_N2.Ha and Ha_N2.N2 and S2.S2 and Hb.Hb 
	#	print i, Ha_N2.Ha_flux, Ha_N2.fluxn2_0, Ha_N2.fluxn2_1, Hb.Hb_flux, S2.flux0, S2.flux1, o3.flux0, o3.flux1, Ha_N2.Ha, Ha_N2.N2, Hb.Hb, S2.S2, o3.O3, PN_cand
		
	
if __name__ == "__main__":
	main()
			
