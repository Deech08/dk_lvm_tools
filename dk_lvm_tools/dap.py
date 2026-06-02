import numpy as np 
from astropy import units as u
from astropy.table import Table, join, vstack
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, Angle

from .dapTableMixin import dapMixin

import re
import os

from tqdm import tqdm
import glob
import seaborn as sns
from functools import partial

lvm_fiber_diameter = 35.3*u.arcsec
lvm_flux_unit = u.def_unit("lvm_flux", 1e-16 * u.erg *u.s**-1 * u.cm**-2 /(lvm_fiber_diameter**2*np.pi))
u.add_enabled_units([lvm_flux_unit])


class dap(dapMixin, Table):
	"""
	Core LVM DAP Class

	Load, view, manipulate, and plot basic results from LVM DAP

	Parameters
	----------
	filename: 'str', optional, must be keyword
		filename of LVM DAP compiled data table
	dap_files: 'str', 'listlike', optional, must be keyword
		if provided, reads lvm dap files directly
	dap_ver: 'str', optional, must be keyword
		if provided, will search $SAS_BASE_DIR for specified DAP version and load all files
	use_multiprocessing: 'bool', optional, must be keyword
		if True, will use mutliprocessing in reading data from file_list or dap_ver search
	verbose: 'bool', optional, must be keyword
		if True, will print more details while loading data
	check_dap_files: 'bool', must be keyword
		if True, will check that DAP files are complete and ready to load or flag bad ones
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
    n_batch: 'int'
    	split loading of files list into specified number of batches 
	**kwargs:
		keywords passed to self.read_DAP_file



	"""

	def __init__(self, filename = None, 
				 dap_files = None, 
				 dap_ver = None, 
		 		 use_multiprocessing = True, 
		 		 verbose = False, 
		 		 check_dap_files = False,
		 		 lite = True,
		 		 eline_list = None,
		 		 eline_quants = None,
		 		 include_primary_elines = False,
		 		 n_batch = 1,
		 		 **kwargs):

		self.pal_colorblind = sns.color_palette("colorblind")

		if use_multiprocessing:
			import multiprocessing

		

		if dap_ver is not None:
			self.sas_base_dir = os.environ["SAS_BASE_DIR"]
			self.path_to_data_dir = "sdsswork/lvm/spectro/analysis/"
			self.dap_ver = dap_ver
			
			pattern = "**/*.dap.fits.gz"

			# === Find all matching FITS files recursively ===
			dap_files = glob.glob(os.path.join(self.sas_base_dir, 
											   self.path_to_data_dir, 
											   self.dap_ver, pattern), 
								  recursive = True)
			if verbose:
				# Print only the number of files found if verbose
				print(f"🔍 Found {len(dap_files)} FITS files.")


		if dap_files is not None:
			if check_dap_files:
				print("Checking DAP files for file integrity...")
				if use_multiprocessing:
					with multiprocessing.Pool() as pool:
						good_dap_files = pool.map(check_DAP_file, dap_files)
				else:
					good_dap_files = list(map(check_DAP_file, tqdm(dap_files)))

				good_dap_files = [fname for fname in good_dap_files if fname is not None]
				if len(good_dap_files)>0:
					print("{0} out of {1} DAP files are ready to read!".format(len(good_dap_files), 
																			   len(dap_files)))
					dap_files = good_dap_files
				else:
					raise ValueError("No DAP Files passed the integrity check!")

			#prepare readding command:

			if lite:
				if eline_quants is None:
					eline_quants = ['flux']



			read_dap_lite = partial(read_DAP_file,lite = lite, 
										  eline_quants = eline_quants, 
										  eline_list = eline_list, 
										  include_primary_elines = include_primary_elines)

			print("Reading DAP Files...")
			if use_multiprocessing:
				#split into batches
				if n_batch == 1:
					with multiprocessing.Pool() as pool:
						tables = pool.map(read_dap_lite, dap_files)
						t = vstack(tables)


				# if n_batch == 2:
				# 	split_ind = int(len(dap_files)/2)
				# 	with multiprocessing.Pool() as pool:
				# 		tables_1 = pool.map(read_dap_lite, dap_files[:split_ind])
				# 		t1 = vstack(tables_1)
				# 	with multiprocessing.Pool() as pool:
				# 		tables_2 = pool.map(read_dap_lite, dap_files[split_ind:])
				# 		t2 = vstack(tables_2)
				# 	t = vstack([t1,t2])
				if n_batch > 1:
					if verbose:
						print("Splitting into {0} batches to load...".format(n_batch))
					split_inds = np.linspace(0,len(dap_files),n_batch+1).astype(int)
					tables = []
					for ell in range(n_batch):
						if verbose:
							print("Starting batch {0}...".format(ell+1))
						with multiprocessing.Pool() as pool:
							tt = pool.map(read_dap_lite, dap_files[split_inds[ell]:split_inds[ell+1]])
						tables.append(vstack(tt))
						if verbose:
							print("Batch {0} of {1} complete.".format(ell+1, n_batch))

					t = vstack(tables)
				coords = SkyCoord(t["ra"], t["dec"], frame = "icrs").transform_to("galactic")
				t["GAL-LON"] = coords.l 
				t["GAL-LAT"] = coords.b 

				super().__init__(data = t.columns, meta = t.meta, **kwargs)
					# del t


			else:
				print("starting to read data...")
				tables = list(map(read_dap_lite, tqdm(dap_files)))

				t = vstack(tables)

				coords = SkyCoord(t["ra"], t["dec"], frame = "icrs").transform_to("galactic")
				t["GAL-LON"] = coords.l 
				t["GAL-LAT"] = coords.b 

				# print(t)


				super().__init__(data = t.columns, meta = t.meta, **kwargs)
				# del t

		elif filename is not None:
			t = Table.read(filename)

			# Try to get DAP version from filename
			v_pattern = r"v_\d+\.\d+\.\d+"
			match = re.search(v_pattern, filename)
			if match:
				self.dap_ver = match.group().split("v")[-1]
			if "GAL-LON" not in t.colnames:
				coords = SkyCoord(t["ra"], t["dec"], frame = "icrs").transform_to("galactic")
				t["GAL-LON"] = coords.l 
				t["GAL-LAT"] = coords.b 

			for colname in t.colnames:
				if t[colname].unit is None:
					if "flux" in colname:
						t[colname].unit = lvm_flux_unit



			super().__init__(data = t.columns, meta = t.meta, **kwargs)
			# del t

		else:
			super().__init__(**kwargs)
		

	def write(self, *args, **kwargs):
		for colname in self.colnames:
			if self[colname].unit == lvm_flux_unit:
				self[colname].unit = None

		super().write(*args, **kwargs)


		

		

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
			"flux":lvm_flux_unit,
			"vel":u.km/u.s,
			"disp":u.AA,
			"EW":u.AA,
		}
	
	with fits.open(dap_file) as dap_hdu:
		tab_PT=Table(dap_hdu['PT'].data)
		if tab_PT["id"].dtype.kind == "S":
			tab_PT["id"] = np.char.decode(tab_PT["id"], 'utf-8', errors = "replace") 
		if include_primary_elines:
			tab_PE=Table(dap_hdu['PM_ELINES'].data)
			if tab_PE["id"].dtype.kind == "S":
				tab_PE["id"] = np.char.decode(tab_PE["id"], 'utf-8', errors = "replace")
		tab_NPE_B=Table(dap_hdu['NP_ELINES_B'].data)
		if tab_NPE_B["id"].dtype.kind == "S":
			tab_NPE_B["id"] = np.char.decode(tab_NPE_B["id"], 'utf-8', errors = "replace")
		tab_NPE_R=Table(dap_hdu['NP_ELINES_R'].data)
		if tab_NPE_R["id"].dtype.kind == "S":
			tab_NPE_R["id"] = np.char.decode(tab_NPE_R["id"], 'utf-8', errors = "replace")
		tab_NPE_I=Table(dap_hdu['NP_ELINES_I'].data)
		if tab_NPE_I["id"].dtype.kind == "S":
			tab_NPE_I["id"] = np.char.decode(tab_NPE_I["id"], 'cp1252', errors = "replace") #don't know why but this stops reading errors...
		if not lite:
			tab_RSP=Table(dap_hdu['RSP'].data)
			if tab_RSP["id"].dtype.kind == "S":
				tab_RSP["id"] = np.char.decode(tab_RSP["id"], 'utf-8', errors = "replace")
			tab_COEFFS=Table(dap_hdu['COEFFS'].data)
			if tab_COEFFS["id"].dtype.kind == "S":
				tab_COEFFS["id"] = np.char.decode(tab_COEFFS["id"], 'utf-8', errors = "replace")
			kel_ext = 0
			try:
				kel_ext = 1
				tab_KEL=Table(dap_hdu['PM_KEL'].data)
				if tab_KEL["id"].dtype.kind == "S":
					tab_KEL["id"] = np.char.decode(tab_KEL["id"], 'utf-8', errors = "replace")
			except:
				kel_ext = 0
				print('No PM_KEL extension')
		
			sig_ext = 0
			try:
				sig_ext = 1
				tab_SIGMA = Table(dap_hdu['ELINES_SIGMA_CHI'].data)
				if tab_SIGMA["id"].dtype.kind == "S":
					tab_SIGMA["id"] = np.char.decode(tab_SIGMA["id"], 'utf-8', errors = "replace")
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