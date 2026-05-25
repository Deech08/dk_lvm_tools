import numpy as np 
import matplotlib.pyplot as plt 
from astropy import units as u 
from astropy.io import fits

from matplotlib.colors import LogNorm

try:
	import cartopy.crs as ccrs
except ModuleNotFoundError:
	pass

from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
from astropy.table import Table
import pandas as pd
import re

class dapMixin(object):
	"""
	Mixin class with convenience functions for LVM DAP data
	"""

	def check_DAP_file(dap_file):
		with fits.open(dap_file) as dap_hdu:
			try:
				tab_NPE_I=Table(dap_hdu['NP_ELINES_I'].data)
			except:
				# print("{} does not have NP_ELINES_I".format(dap_file))
				return None
			else:
				return dap_file

	# based on code from https://github.com/sdss/lvmdap/blob/main/lvmdap/dap_tools.py
	def read_DAP_file(dap_file,
					  verbose=False, 
					  lite = False, 
					  eline_list = None, 
					  eline_quants = None, 
					  include_primary_elines = True):
		"""
		Read DAP file and extract data as an astropy Table

		PARAMETERS
		----------
		dap_file: `str`
			filename of DAP file to load
		verbose: `bool`
			print verbose messages during process, if True
		lite: `bool`
			if True, only loads limited data
		eline_list: `list-like`
			only used if lite is True
			list of emission line columns to extract
			if lite is True, uses a default list of the following:
		eline_quants: `list-like`
			only used if lite is True
			list of emission line quantities that have been measured to extract
			if lite is True, uses a default list of the following:
			[
			"flux",
			"vel",
			"disp",
			"EW",
			]
		include_primary_elines: `bool`
			if True, includes primary emission line measurements from Gaussian Model
			Default is True
			
		"""
		if lite:
			if eline_list is None:
				# set default emission line list
				eline_list = [
					"[OII]_3726.03", 
					"[OII]_3728.82", 
					"Hepsilon_3970.07",
					"Hdelta_4101.77",
					"Hgamma_4340.49",
					"Hbeta_4861.36",
					"[OIII]_5006.84",
					"[OI]_5577.34",
					"[NII]_5754.59",
					"[OI]_6300.3",
					"[SIII]_6312.06",
					"[NII]_6548.05",
					"Halpha_6562.85",
					"[NII]_6583.45",
					"[SII]_6716.44",
					"[SII]_6730.82",
					"[SIII]_9531.1",
				]
			if eline_quants is None:
				print("yes")
				eline_quants = [
					"flux",
					"vel",
					"disp",
					"EW"
				]

			eline_list = np.array(eline_list)
			eline_list_wavs = np.array([float(line.split("_")[-1]) for line in eline_list])
			blue = eline_list_wavs < 5755.
			infrared = eline_list_wavs > 7637.
			red = (~blue) & (~infrared)
			

			unit_dict = {
				"flux":lvm_flux,
				"vel":u.km/u.s,
				"disp":u.AA,
				"EW":u.AA,
			}
		
		with fits.open(dap_file) as dap_hdu:
			tab_PT=Table(dap_hdu['PT'].data)
			if include_primary_elines:
				tab_PE=Table(dap_hdu['PM_ELINES'].data)
			tab_NPE_B=Table(dap_hdu['NP_ELINES_B'].data)
			tab_NPE_R=Table(dap_hdu['NP_ELINES_R'].data)
			tab_NPE_I=Table(dap_hdu['NP_ELINES_I'].data)
			if not lite:
				tab_RSP=Table(dap_hdu['RSP'].data)
				tab_COEFFS=Table(dap_hdu['COEFFS'].data)
				kel_ext = 0
				try:
					kel_ext = 1
					tab_KEL=Table(dap_hdu['PM_KEL'].data)
				except:
					kel_ext = 0
					print('No PM_KEL extension')
			
				sig_ext = 0
				try:
					sig_ext = 1
					tab_SIGMA = Table(dap_hdu['ELINES_SIGMA_CHI'].data)
				except:
					sig_ext = 0
					print('No SIGMA_CHI extension')

		if not lite:
			# Add units for RSP
			tab_RSP["Teff"] *= u.K
			tab_RSP["e_Teff"] *= u.K
			tab_RSP["disp"] *= u.km/u.s
			tab_RSP["e_disp"] *= u.km/u.s
			tab_RSP["flux"] *= lvm_flux
			tab_RSP["med_flux"] *= lvm_flux
			tab_RSP["e_med_flux"] *= lvm_flux
			tab_RSP["Teff_MW"] *= u.K
			tab_RSP["e_Teff_MW"] *= u.K
			tab_RSP["sys_vel"] *= u.km/u.s
			
			#
			# Rename some entries!
			#
			tab_RSP.rename_column('Av','Av_st')
			tab_RSP.rename_column('e_Av','e_Av_st')
			tab_RSP.rename_column('z','z_st')
			tab_RSP.rename_column('e_z','e_z_st')
			tab_RSP.rename_column('disp','disp_st')
			tab_RSP.rename_column('e_disp','e_disp_st')
			tab_RSP.rename_column('flux','flux_st')
			tab_RSP.rename_column('redshift','redshift_st')
			tab_RSP.rename_column('med_flux','med_flux_st')
			tab_RSP.rename_column('e_med_flux','e_med_flux_st')
			tab_RSP.rename_column('sys_vel','vel_st')

			if (kel_ext == 1):
				# Add units
				tab_KEL["flux"] *= lvm_flux
				tab_KEL["e_flux"] *= lvm_flux
				tab_KEL["disp"] *= u.AA
				tab_KEL["e_disp"] *= u.AA
				tab_KEL["vel"] *= u.km/u.s
				tab_KEL["e_vel"] *= u.km/u.s
				
				tab_KEL.rename_column('flux','flux_pek')
				tab_KEL.rename_column('e_flux','e_flux_pek')
				tab_KEL.rename_column('disp','disp_pek')
				tab_KEL.rename_column('e_disp','e_disp_pek')
				tab_KEL.rename_column('vel','vel_pek')
				tab_KEL.rename_column('e_vel','e_vel_pek')
		
				
			#
			# id    id_fib  rsp TEFF    LOGG    META    ALPHAM  COEFF   Min.Coeff   log(M/L)    AV  N.Coeff Err.Coeff
			#
			# Add units
			tab_COEFFS["TEFF"] *= u.K
			
			tab_COEFFS.rename_column('rsp','id_rsp')
			tab_COEFFS.rename_column('TEFF','Teff_rsp')
			tab_COEFFS.rename_column('LOGG','Log_g_rsp')
			tab_COEFFS.rename_column('META','Fe_rsp')
			tab_COEFFS.rename_column('ALPHAM','alpha_rsp')
			tab_COEFFS.rename_column('COEFF','W_rsp')
			tab_COEFFS.rename_column('Min.Coeff','min_W_rsp')
			tab_COEFFS.rename_column('log(M/L)','log_ML_rsp')
			tab_COEFFS.rename_column('AV','Av_rsp')
			tab_COEFFS.rename_column('N.Coeff','n_W_rsp')
			tab_COEFFS.rename_column('Err.Coeff','e_W_rsp')
		if include_primary_elines:
			#
			# Parametric elines
			#
			
			# Add units
			tab_PE["flux"] *= lvm_flux
			tab_PE["e_flux"] *= lvm_flux
			tab_PE["disp"] *= u.AA
			tab_PE["e_disp"] *= u.AA
			tab_PE["vel"] *= u.km/u.s
			tab_PE["e_vel"] *= u.km/u.s
			
			tab_PE.rename_column('flux','flux_pe')
			tab_PE.rename_column('e_flux','e_flux_pe')
			tab_PE.rename_column('disp','disp_pe')
			tab_PE.rename_column('e_disp','e_disp_pe')
			tab_PE.rename_column('vel','vel_pe')
			tab_PE.rename_column('e_vel','e_vel_pe')

		
		
		tab_DAP=tab_PT[:]

		# Add units for ra/dec
		tab_DAP["ra"] *= u.deg
		tab_DAP["dec"] *= u.deg
		
		if not lite:
			tab_DAP=join(tab_DAP,tab_RSP,keys=['id'],join_type='left')
			# Units for B
			for name in tab_NPE_B.colnames:
				if "flux_" in name:
					tab_NPE_B[name] *= lvm_flux
				elif "vel_" in name:
					tab_NPE_B[name] *= u.km/u.s
				elif "disp_" in name:
					tab_NPE_B[name] *= u.AA
				elif "EW_" in name:
					tab_NPE_B[name] *= u.AA
			tab_DAP=join(tab_DAP,tab_NPE_B,keys=['id'],join_type='left')
			# Units for R
			for name in tab_NPE_R.colnames:
				if "flux_" in name:
					tab_NPE_R[name] *= lvm_flux
				elif "vel_" in name:
					tab_NPE_R[name] *= u.km/u.s
				elif "disp_" in name:
					tab_NPE_R[name] *= u.AA
				elif "EW_" in name:
					tab_NPE_R[name] *= u.AA
			tab_DAP=join(tab_DAP,tab_NPE_R,keys=['id'],join_type='left')
			# Units for I
			for name in tab_NPE_I.colnames:
				if "flux_" in name:
					tab_NPE_I[name] *= lvm_flux
				elif "vel_" in name:
					tab_NPE_I[name] *= u.km/u.s
				elif "disp_" in name:
					tab_NPE_I[name] *= u.AA
				elif "EW_" in name:
					tab_NPE_I[name] *= u.AA
			tab_DAP=join(tab_DAP,tab_NPE_I,keys=['id'],join_type='left')
		else:
			# Units for B
			blue_names = ["id"]
			for line in eline_list[blue]:
				for quant in eline_quants:
					name = "{}_{}".format(quant, line)
					ename = "e_{}".format(name)
					tab_NPE_B[name] *= unit_dict[quant]
					tab_NPE_B[ename] *= unit_dict[quant]
					blue_names.append(name)
					blue_names.append(ename)
			tab_DAP=join(tab_DAP,tab_NPE_B[blue_names],keys=['id'],join_type='left')
			# Units for R
			red_names = ["id"]
			for line in eline_list[red]:
				for quant in eline_quants:
					name = "{}_{}".format(quant, line)
					ename = "e_{}".format(name)
					tab_NPE_R[name] *= unit_dict[quant]
					tab_NPE_R[ename] *= unit_dict[quant]
					red_names.append(name)
					red_names.append(ename)
			tab_DAP=join(tab_DAP,tab_NPE_R[red_names],keys=['id'],join_type='left')
			# Units for I
			infrared_names = ["id"]
			for line in eline_list[infrared]:
				for quant in eline_quants:
					name = "{}_{}".format(quant, line)
					ename = "e_{}".format(name)
					tab_NPE_I[name] *= unit_dict[quant]
					tab_NPE_I[ename] *= unit_dict[quant]
					infrared_names.append(name)
					infrared_names.append(ename)
			tab_DAP=join(tab_DAP,tab_NPE_I[infrared_names],keys=['id'],join_type='left')

		if include_primary_elines:
			#
			# order parametric emission line table
			#
			mask_elines = (tab_PE['model']=='eline')
			tab_PE = tab_PE[mask_elines]
			
			a_wl = np.unique(tab_PE['wl'])
			I=0
			for wl_now in a_wl:
				if (wl_now>0.0):
					tab_PE_now=tab_PE[tab_PE['wl']==wl_now]
					tab_PE_tmp=tab_PE_now['id','flux_pe','e_flux_pe','disp_pe','e_disp_pe','vel_pe','e_vel_pe']
					for cols in tab_PE_tmp.colnames:        
						if (cols != 'id'):
							tab_PE_tmp.rename_column(cols,f'{cols}_{wl_now}')
					if (I==0):
						tab_PE_ord=tab_PE_tmp
					else:
						tab_PE_ord=join(tab_PE_ord,tab_PE_tmp,keys=['id'],join_type='left')
					I=I+1        
			tab_DAP=join(tab_DAP,tab_PE_ord,keys=['id'],join_type='left')

		if not lite:
			#
			# order parametric emission line table with fixed kinematics
			#
			if (kel_ext == 1):
				mask_elines = (tab_KEL['model']=='eline')
				tab_KEL = tab_KEL[mask_elines]
			
				a_wl = np.unique(tab_KEL['wl'])
				I=0
				for wl_now in a_wl:
					if (wl_now>0.0):
						tab_KEL_now=tab_KEL[tab_KEL['wl']==wl_now]
						tab_KEL_tmp=tab_KEL_now['id','flux_pek','e_flux_pek','disp_pek','e_disp_pek','vel_pek','e_vel_pek']
						for cols in tab_KEL_tmp.colnames:        
							if (cols != 'id'):
								tab_KEL_tmp.rename_column(cols,f'{cols}_{wl_now}')
						if (I==0):
							tab_KEL_ord=tab_KEL_tmp
						else:
							tab_KEL_ord=join(tab_KEL_ord,tab_KEL_tmp,keys=['id'],join_type='left')
						I=I+1        
				tab_DAP=join(tab_DAP,tab_KEL_ord,keys=['id'],join_type='left')
		
			#
			# Order COEFFS table
			#
			a_rsp=np.unique(tab_COEFFS['id_rsp'])
			for I,rsp_now in enumerate(a_rsp):
				tab_C_now=tab_COEFFS[tab_COEFFS['id_rsp']==rsp_now]
				tab_C_tmp=tab_C_now['id','Teff_rsp', 'Log_g_rsp', 'Fe_rsp',\
									'alpha_rsp', 'W_rsp', 'min_W_rsp',\
									'log_ML_rsp', 'Av_rsp', 'n_W_rsp', 'e_W_rsp']
				for cols in tab_C_tmp.colnames:        
					if (cols != 'id'):
						tab_C_tmp.rename_column(cols,f'{cols}_{rsp_now}')
				if (I==0):
					tab_C_ord=tab_C_tmp
				else:
					tab_C_ord=join(tab_C_ord,tab_C_tmp,keys=['id'],join_type='left')
			tab_DAP=join(tab_DAP,tab_C_ord,keys=['id'],join_type='left')
		
			if (sig_ext == 1):
				tab_DAP=join(tab_DAP,tab_SIGMA,keys=['id'],join_type='left')
		 
			if (verbose==True):
				print('---- ALL Table Columns -----')
				print('-------------------------------')
				print('|        PT                   |')
				print('-------------------------------')
				list_columns(tab_PT.colnames)
				print('----------------------------------')
				print('|        RSP                      |')
				print('----------------------------------')
				list_columns(tab_RSP.colnames)
				print('----------------------------------')
				print('|        PE_ord                   |')
				print('----------------------------------')
				list_columns(tab_PE_ord.colnames)
				if (kel_ext == 1):
					print('----------------------------------')
					print('|        PEK_ord                   |')
					print('----------------------------------')
					list_columns(tab_KEL_ord.colnames)
				print('----------------------------------')
				print('|        NPE_B                    |')
				print('----------------------------------')
				list_columns(tab_NPE_B.colnames,3)
				print('----------------------------------')
				print('|        NPE_R                    |')
				print('----------------------------------')
				list_columns(tab_NPE_R.colnames,3)
				print('----------------------------------')
				print('|        NPE_I                    |')
				print('----------------------------------')
				list_columns(tab_NPE_I.colnames,3)
				print('----------------------------------')
				print('|        C_ord                    |')
				print('----------------------------------')
				list_columns(tab_C_ord.colnames,4)
				if (sig_ext == 1):
					print('----------------------------------')
					print('|        SIGMA_CHI                   |')
					print('----------------------------------')
					list_columns(tab_SIGMA.colnames)
			
		return tab_DAP


	def plot_line_ratios_sii_nii_vs_halpha(self,
											line_prefix='flux_',
											wave_halpha=6563.0,
											wave_nii=6584.0,
											wave_sii=6716.0,
											ra_col='ra',
											dec_col='dec',
											kind='hex',
											bins=50,
											scaling_factor=None,
											min_abs_b_deg=7,
											max_abs_b_deg=60,
											stretch='linear',
											power=0.3,
											save_dir=None,
											show_plots=False,
											histlog = True,
											max_xval = 1.2,
											max_yval = 1.2,
											**kwargs
											):
		"""
		Plot [S II]/Halpha vs [N II]/Halpha with model lines for temperature and sulfur ionization

		Parameters
		----------
		kind: 'str',
			kind of plot to make ["scatter", "hex", etc]
		bins: 'int'
			number of bins to use for histogramming
		scaling_factor:

		min_abs_b_deg: 'float',
			minimum absolute value of latitude to consider, default to 7 degrees
		max_abs_b_def: 'float'
			maximum absolute value of latitude to consider, default to 60 degrees
		stretch: 'str'
			stretch kind to use ??
		power: 'float'
			???
		save_dir: 'str'
			directory to save figure to if provided. creates directory if doesn't exist
			if None, will not auto save figure
		histlog: 'bool'
			if True, will log scale 1D histograms 
		max_xval: 'float'
			max value to consider for [N II]/Halpha axis
		max_xval: 'float'
			max value to consider for [S II]/Halpha axis
		"""
		# set up
		line_prefix='flux_',
		wave_halpha=6563.0,
		wave_nii=6584.0,
		wave_sii=6716.0,
		ra_col='ra',
		dec_col='dec',
		
		# Create output directory if needed
		if save_dir is not None:
			os.makedirs(save_dir, exist_ok=True)

		# --- Find flux columns by wavelength ---
		pattern = re.compile(r'_(\d+\.\d+)$')
		flux_cols = {
			float(pattern.search(col).group(1)): col
			for col in self.colnames
			if col.startswith(line_prefix) and pattern.search(col)
		}

		if not flux_cols:
			print("❌ No flux columns found with expected pattern.")
			return

		def find_closest_wave(target, options):
			return min(options, key=lambda w: abs(w - target))

		available_waves = list(flux_cols.keys())
		col_halpha = flux_cols[find_closest_wave(wave_halpha, available_waves)]
		col_nii = flux_cols[find_closest_wave(wave_nii, available_waves)]
		col_sii = flux_cols[find_closest_wave(wave_sii, available_waves)]
		# col_sii_6731 = flux_cols[find_closest_wave(wave_sii_6731, available_waves)]

		print("🔍 Matched Emission Lines:")
		print(f"  Hα:   {col_halpha}")
		print(f"  [N II]: {col_nii}")
		print(f"  [S II] 6716: {col_sii}")
		# print(f"  [S II] 6731: {col_sii_6731}")

		# --- Galactic Latitude Filter ---
		coords = SkyCoord(ra=np.asarray(self[ra_col], dtype=float) * u.deg,
						  dec=np.asarray(self[dec_col], dtype=float) * u.deg,
						  frame='icrs')
		b_deg = coords.galactic.b.deg
		lat_mask = (np.abs(b_deg) >= min_abs_b_deg) & (np.abs(b_deg) <= max_abs_b_deg)
		print(f"📌 {np.sum(lat_mask)} sources within |b| ∈ [{min_abs_b_deg}, {max_abs_b_deg}]°")

		# --- Extract fluxes and convert to Rayleighs ---
		h = 6.626e-27  # erg·s
		c = 2.998e18   # Å/s
		rayleigh_factor = 1e6 / (4 * np.pi)

		def to_rayleighs(flux, wave):
			photons = (flux * wave) / (h * c)
			return photons / rayleigh_factor

		wave_halpha_actual = float(re.search(r'_(\d+\.\d+)$', col_halpha).group(1))
		wave_nii_actual = float(re.search(r'_(\d+\.\d+)$', col_nii).group(1))
		wave_sii_actual = float(re.search(r'_(\d+\.\d+)$', col_sii).group(1))
		# wave_sii_6731_actual = float(re.search(r'_(\d+\.\d+)$', col_sii_6731).group(1))

		flux_halpha = np.asarray(self[col_halpha], dtype=float)
		flux_nii = np.asarray(self[col_nii], dtype=float)
		flux_sii = np.asarray(self[col_sii], dtype=float)
		# flux_sii_6731 = np.asarray(self[col_sii_6731], dtype=float)

		ray_halpha = to_rayleighs(flux_halpha, wave_halpha_actual)
		ray_nii = to_rayleighs(flux_nii, wave_nii_actual)
		ray_sii = to_rayleighs(flux_sii, wave_sii_actual)
		# ray_sii_6731 = to_rayleighs(flux_sii_6731, wave_sii_6731_actual)
		# ray_sii = ray_sii_6716 + ray_sii_6731

		if scaling_factor:
			ray_halpha *= scaling_factor
			ray_nii *= scaling_factor
			ray_sii *= scaling_factor

		# --- Compute ratios ---
		nii_halpha = ray_nii / ray_halpha
		sii_halpha = ray_sii / ray_halpha

		# --- Apply latitude mask and clean ---
		df = pd.DataFrame({
			'nii_halpha': nii_halpha[lat_mask],
			'sii_halpha': sii_halpha[lat_mask]
		}).replace([np.inf, -np.inf], np.nan).dropna()

		df = df[(df > 0).all(axis=1)]
		#filter out values over max for speed up
		df = df[(df < max(max_xval,max_yval)).all(axis=1)]
		if df.empty:
			print("⚠️ No valid data after filtering.")
			return

		# --- Apply value stretch ---
		label_suffix = ''
		if stretch == 'log':
			df = np.log10(df)
			label_suffix = ' (log10)'
		elif stretch == 'sqrt':
			df = np.sqrt(df)
			label_suffix = ' (sqrt)'
		elif stretch == 'power':
			df = np.power(df, power)
			label_suffix = f' (power={power})'
		elif stretch != 'linear':
			raise ValueError(f"Unsupported stretch mode: {stretch}")

		# --- Clip outliers ---
		for col in df.columns:
			high = np.percentile(df[col], 99.9)
			df = df[df[col] <= high]

		print(f"📈 Plotting {len(df)} points")

		# --- Plotting Layout ---
		fig = plt.figure(figsize=(8, 8))
		gs = fig.add_gridspec(4, 4)
		ax_main = fig.add_subplot(gs[:3, :3])
		ax_xhist = fig.add_subplot(gs[3, :3])
		ax_yhist = fig.add_subplot(gs[:3, 3])

		if kind == 'hex':
			cmap = kwargs.pop('cmap', 'plasma')
			hb = ax_main.hexbin(
				df['nii_halpha'], df['sii_halpha'],
				gridsize=bins,
				cmap=cmap,
				mincnt=1,
				bins='log',
				**kwargs
			)
			# fig.colorbar(hb, ax=ax_main, label='log10(Counts)')
		elif kind == 'scatter':
			ax_main.scatter(
				df['nii_halpha'], df['sii_halpha'],
				s=kwargs.get('s', 5),
				c=kwargs.get('c', 'blue'),
				alpha=kwargs.get('alpha', 0.3),
				edgecolor='none'
			)

		# --- Overlay model curves and lines ---
		tlevels = np.linspace(5000, 9000, 5)
		for i, tempy in enumerate(tlevels):
			t4 = tempy / 1e4
			n2val = 1.62e5 * t4 * np.exp(-2.18 / t4) * 0.8 * 7.5e-5
			ax_main.axvline(n2val, c='w', ls='--')
			ax_main.text(n2val + 0.02, 0.6, f'{int(tempy)} K', color='w',
						 verticalalignment='center', rotation=90, fontsize = 12)

		tlevels = np.linspace(0.25, 0.75, 3)
		for i, tlevels in enumerate(tlevels):
			grad = 4.62 * tlevels * 1.3e-5 / (7.5e-5 * 0.8)
			xvals = np.linspace(0, max_xval, 100)
			yvals = xvals * grad
			ax_main.plot(xvals, yvals, c='b', ls='--')

		ax_main.text(0.6, 0.21, f'{0.25}', color='b', verticalalignment='center', rotation=25, fontsize = 12)
		ax_main.text(0.8, 0.45, f'{0.5}', color='b', verticalalignment='center', rotation=40, fontsize = 12)
		ax_main.text(0.8, 0.7, f'{0.75}', color='b', verticalalignment='center', rotation=45, fontsize = 12)

		# --- Axis Labels & Histograms ---
		ax_main.set_xlabel(f'[N II]/Hα{label_suffix}', fontsize = 12)
		ax_main.set_xlim(0,max_xval)
		ax_main.set_ylabel(f'[S II]/Hα{label_suffix}', fontsize = 12)
		ax_main.set_ylim(0,max_yval)
		ax_main.set_title(f'Emission Line Ratios vs. Hα\n|b| ∈ [{min_abs_b_deg}°, {max_abs_b_deg}°]')
		ax_main.grid(alpha=0.3)

		ax_xhist.hist(df['nii_halpha'], bins=bins, color='gray', alpha=0.6, log = histlog)
		ax_xhist.set_ylabel("Count")
		ax_xhist.grid(alpha=0.3)

		ax_yhist.hist(df['sii_halpha'], bins=bins, color='gray', alpha=0.6, orientation='horizontal', log = histlog)
		ax_yhist.set_xlabel("Count")
		ax_yhist.grid(alpha=0.3)

		plt.setp(ax_xhist.get_xticklabels(), visible=False)
		plt.setp(ax_yhist.get_yticklabels(), visible=False)

		fig.tight_layout()
		fig.subplots_adjust(hspace=0.5, wspace=0.5)

		return fig

		if save_dir is not None:
			fname = f'sii_nii_vs_halpha_{kind}_{stretch}_ [{min_abs_b_deg}°, {max_abs_b_deg}°].png'
			path = os.path.join(save_dir, fname)
			fig.savefig(path, dpi = 300, transparent = True)
			print(f"✅ Saved: {path}")

		if show_plots:
			plt.show()
			plt.close(fig)

		
