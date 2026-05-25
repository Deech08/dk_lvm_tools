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
		 		 **kwargs):

		self.lvm_flux_unit = u.def_unit("lvm_flux", 1e-16 * u.erg *u.s**-1 * u.cm**-2)
		u.add_enabled_units([self.lvm_flux_unit])

		if dap_ver is not None:
			self.sas_base_dir = os.environ["SAS_BASE_DIR"]
			self.path_to_dat_dir = "sdsswork/lvm/spectro/analysis/"
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
						good_dap_files = pool.map(self.check_DAP_file, dap_files)
				else:
					good_dap_files = list(map(self.check_DAP_file, tqdm(dap_files)))

				good_dap_files = [fname for fname in good_dap_files if fname is not None]
				if len(good_dap_files)>0:
					print("{0} out of {1} DAP files are ready to read!".format(len(good_dap_files, 
																			   len(dap_files))))
					dap_files = good_dap_files
				else:
					raise ValueError("No DAP Files passed the integrity check!")

			#prepare readding command:

			if lite:
				if eline_quants is None:
					eline_quants = ['flux']



			def read_dap_lite(dap_file):
				return self.read_DAP_file(dap_file, 
										  lite = lite, 
										  eline_quants = eline_quants, 
										  eline_list = eline_list, 
										  include_primary_elines = include_primary_elines)

			print("Reading DAP Files...")
			if use_multiprocessing:
				tables = pool.map(read_dap_lite, dap_files)
			else:
				tables = list(map(read_dap_lite, tqdm(dap_files)))

			t = vstack(tables)




		elif filename is not None:
			t = Table.read(filename)

			# Try to get DAP version from filename
			v_pattern = r"v_\d+\.\d+\.\d+"
			match = re.search(v_pattern, filename)
			if match:
				self.dap_ver = match.group().split("v")[-1]



		super().__init__(data = t.columns, meta = t.meta)

		del t

