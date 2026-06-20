import numpy as np 
import matplotlib.pyplot as plt 
from astropy import units as u 
from astropy.io import fits
import astropy.constants as constants

from matplotlib.colors import LogNorm, Normalize

try:
    import cartopy.crs as ccrs
except ModuleNotFoundError:
    pass

from astropy.coordinates import SkyCoord, LSR
from astropy.coordinates import Angle
from astropy.table import Table
import pandas as pd
import re
import seaborn as sns
import logging

from scipy.interpolate import griddata, interp1d
from astropy.utils.masked import Masked
from scipy.ndimage import gaussian_filter

lvm_fiber_diameter = 35.3*u.arcsec
lvm_flux_unit = u.def_unit("lvm_flux", 1e-16 * u.erg *u.s**-1 * u.cm**-2 /(lvm_fiber_diameter**2*np.pi))
u.add_enabled_units([lvm_flux_unit])

class dapMixin(object):
    """
    Mixin class with convenience functions for LVM DAP data
    """

    def __get_item__(self, item):
        sliced_super = super().__get_item__(item)

        return self.__class__(data = sliced_super.data)

    def load_drpall(self, filename = None):
        """
        Loads drp all file as an astropy Table

        Parameters
        ----------
        filename: 'str', optional
            if provided reads specified file
            defaults to default sas path
        """

        if filename is None:
            filename = os.path.join(self.sas_base_dir, 
                                    "sdsswork/lvm/spectro/redux", 
                                    self.dap_ver, 
                                    "drpall-{}.fitss".format(self.dap_ver))

        self.drp_all = Table.read(filename)

    def load_obstimes(self):
        """
        Loads obstimes from drpall
        """
        if self.drpall is None:
            self.load_drpall
        obstime_lookup_table = {
            "{}".format(x):y for x,y in  zip(self.drpall["expnum"],self.drpall["obstime"])
        }
        expnums = np.array([x.split(".")[0] for x in self["id"].astype(str)]).astype(int)
        self["obstime"] = [obstime_lookup_table["{}".format(expnum)] for expnum in expnums]



    def convert_flux_to_rayleigh(self, colname):
        """
        Converts specified column of flux values to photon units from energy units

        Parameters
        ----------
        colname: 'str'
            name of column to convert flux values of



        """
        # --- Extract fluxes and convert to Rayleighs ---
        h = constants.h
        c = constants.c

        wave = float(colname.split("_")[-1])*u.AA 
        line_energy  = h * c / wave 
        photon_flux = self[colname] / line_energy * u.photon
        return(photon_flux.to(u.rayleigh))





    def plot_histogram(self, x, y = None, ax = None, fig = None, snr_cut = 3, 
        data_mask = None, label_axes = True, log_scale = False, rayleigh = False, **kwargs):
        """
        Uses Seaborns displot on columns of table

        Parameters
        ----------
        x: 'str'
            name of x-axis column to consider
        y: 'str'
            name of y-axis column to consider
        data_mask: 'list-like'
            custom masking array to apply to data
        snr_cut: 'float'
            SNR to cut data by set to 0 to not use SNR info (for velocities, etc.)
        data_mask: 'np.array'
            mask to use on the specified columns
        label_axes: 'bool'
            if True, adds x and y axis labels
        log_scale: 'bool', list-like
            if True, will log scale x values
            if 2 len list-like, will apply log scale to x and/or y axis [x, y]
        rayleigh: 'bool':
            if True, convert intensity to Rayleigh - only for flux!
        kwargs:
            passed to ax.hexbin 
        """

        if not hasattr(ax, 'scatter'):
            if not hasattr(fig, 'add_subplot'):
                fig = plt.figure(constrained_layout = True)
                ax = fig.add_subplot(111)
            else:
                ax = fig.add_subplot(111)
        else:
            if not hasattr(fig, 'add_subplot'):
                fig = ax.get_figure()

        if snr_cut == 0.:
            snr_mask = np.ones_like(self[x], dtype = bool)
        snr_x = self[x]/self["e_{}".format(x)]
        snr_mask = snr_x>snr_cut
        if y is not None:
            snr_y = self[y]/self["e_{}".format(y)]
            snr_mask &= snr_y>snr_cut

        if data_mask is not None:
            snr_mask &= data_mask

        if log_scale.__class__ is not bool:
            logx, logy = log_scale
        else:
            logx = log_scale
            logy = False

        if logx:
            if rayleigh:
                xx = np.log10(self.convert_flux_to_rayleigh(x).value[snr_mask])
            else:
                xx = np.log10(self[x][snr_mask])
            x_suffix = "(log_10)"
        else:
            if rayleigh:
                xx = self.convert_flux_to_rayleigh(x)[snr_mask]
            else:
                xx = self[x][snr_mask]
            x_suffix = ""

        if y is not None:
            if logy:
                if rayleigh:
                    yy = np.log10(self.convert_flux_to_rayleigh(y).value[snr_mask])
                else:
                    yy = np.log10(self[y][snr_mask])
                y_suffix = "(log_10)"
            else:
                if rayleigh:
                    yy = self.convert_flux_to_rayleigh(y)[snr_mask]
                else:
                    yy = self[y][snr_mask]
                y_suffix = ""

            hb = ax.hexbin(xx, yy, bins = kwargs.pop("bins","log"), 
                            gridsize = kwargs.pop("gridsize",1000),
                            mincnt = kwargs.pop("mincnt",1), 
                            **kwargs)
            if label_axes:
                ax.set_xlabel("{} {}".format(x, x_suffix), fontsize = 12)
                ax.set_ylabel("{} {}".format(y, y_suffix), fontsize = 12)

        else:
            hb = ax.hist(xx, bins = kwargs.pop("bins",50), label = x, **kwargs)
            if label_axes:
                ax.set_xlabel("{} {}".format(x, x_suffix), fontsize = 12)
                ax.set_ylabel("Counds", fontsize = 12)

        return fig




    def get_snr_masked_intensity(self, colname, snr_cut = 3, rayleigh = True, scale = None, velocity = False):
        """
        Get Masked Array of intensity at given SNR cut

        Parameters
        ----------
        colname: 'str'
            name of column to get masked array for
        snr_cut: 'float'
            threshold for SNR cut to use
        rayleigh: 'bool'
            if True, returns flux in photon units rather than default energy units
        scale:'str'
            'log' or 'log10' to scale intensity and remove units
        velocity: 'bool'
            if True, also returns velocity of same line with same SNR cut on flux

        Returns
        -------
        masked_flux = numpy.ma.array of flux masked to specified SNR threshold
        """

        if rayleigh:
            flux = self.convert_flux_to_rayleigh(colname)
            error = self.convert_flux_to_rayleigh("e_{}".format(colname))
        else:
            flux = self[colname]
            error = self["e_{}".format(colname)]
        snr = flux/error 

        if scale is None:
            flux_out = Masked(flux, mask = snr < snr_cut|np.isnan(snr))
        elif scale == "log":
            flux_out = Masked(np.log(flux.value), mask = snr < snr_cut|np.isnan(snr))
        elif scale == "log10":
            flux_out = Masked(np.log10(flux.value), mask = snr < snr_cut|np.isnan(snr))

        vel_colname = "vel{}".format(colname[4:])
        vel_out = Masked(self[vel_colname].quantity, mask = flux_out.mask)

        if velocity:
            return flux_out, vel_out
        else:
            return flux_out

    def get_SkyCoord(self, galactic = True):
        """
        Get SkyCoords object for full table

        Parameters
        ----------
        galactic: 'bool'
            if True, returns SkyCoord in Galactic Frame instead of ICRS 

        """
        if not galactic:
            return SkyCoord(ra = self["ra"], dec = self["dec"], frame = "icrs")
        else:
            return SkyCoord(ra = self["ra"], dec = self["dec"], frame = "icrs").transform_to("galactic")

    def get_line_ratio(self, line_num, line_denom, snr_cut = 3, rayleigh = False):
        """
        Get Emission line ratio of specified columns

        Parameters
        ----------
        line_num: 'str'
            numerator column name
        line_denom: 'str'
            denomenator column name
        snr_cut: 'float'
            SNR cut to use
        rayleigh: 'bool'
            if True, computes line ratio in photon units
            if False, computes line ratio in energy units
        """

        flux_num = self.get_snr_masked_intensity(line_num, snr_cut = snr_cut, rayleigh = rayleigh)
        flux_denom = self.get_snr_masked_intensity(line_denom, snr_cut = snr_cut, rayleigh = rayleigh)
        return flux_num / flux_denom

    def get_line_sum(self, line_list, snr_cut = 3, rayleigh = False):
        """
        Get sum of Emission lines

        Parameters
        ----------
        line_list: 'str'
            list of column names to sum
        snr_cut: 'float'
            SNR cut to use
        rayleigh: 'bool'
            if True, computes sum in photon units
            if False, computes sum in energy units
        """

        lines = [self.get_snr_masked_intensity(x, snr_cut = snr_cut, rayleigh = rayleigh) for x in line_list]
        return np.sum(lines, axis = 0)

    def kewley_sf_nii(self,log_nii_ha):
        """Star forming classification line for log([NII]/Ha)."""
        return 0.61 / (log_nii_ha - 0.05) + 1.3


    def kewley_sf_sii(self,log_sii_ha):
        """Star forming classification line for log([SII]/Ha)."""
        return 0.72 / (log_sii_ha - 0.32) + 1.3


    def kewley_sf_oi(self,log_oi_ha):
        """Star forming classification line for log([OI]/Ha)."""
        return 0.73 / (log_oi_ha + 0.59) + 1.33


    def kewley_comp_nii(self,log_nii_ha):
        """Composite classification line for log([NII]/Ha)."""
        return 0.61 / (log_nii_ha - 0.47) + 1.19


    def kewley_agn_sii(self,log_sii_ha):
        """Seyfert/LINER classification line for log([SII]/Ha)."""
        return 1.89 * log_sii_ha + 0.76


    def kewley_agn_oi(self,log_oi_ha):
        """Seyfert/LINER classification line for log([OI]/Ha)."""
        return 1.18 * log_oi_ha + 1.30

    def schawinski_liner_nii(self,log_nii_ha):
        """LINER/AGN Classification line for log([NII]/Ha)"""
        return 1.05 * log_nii_ha + 0.45


    def get_bpt_data(self, nii = True, sii = False, oi = False, snr_cut = 3):
        """
        Computes all line ratios for use in a BPT diagram

        Parameters
        ----------
        nii: 'bool'
            if True, returns [NII] based BPT diagram data under dictionary key "nii"
        sii: 'bool'
            if True, returns [SII] based BPT diagram data under dictionary key "sii"
        oi: 'bool'
            if True, returns [OI] based BPT diagram data under dictionary key "oi"
        snr_cut: 'float'
            SNR cut to use
        """
        #make sure at least one is True
        if not np.any([nii, sii, oi]):
            raise ValueError("at least one of either nii, sii, or oi must be True!")

        # Get necessary emission lines:
        oiii = self.get_snr_masked_intensity("flux_[OIII]_5006.84", rayleigh = False, snr_cut = snr_cut)
        hb = self.get_snr_masked_intensity("flux_Hbeta_4861.36", rayleigh = False, snr_cut = snr_cut)
        ha = self.get_snr_masked_intensity("flux_Halpha_6562.85", rayleigh = False, snr_cut = snr_cut)
        if nii:
            ntwo = self.get_snr_masked_intensity("flux_[NII]_6583.45", rayleigh = False, snr_cut = snr_cut)
        if sii:
            stwo = self.get_line_sum(["flux_[SII]_6716.44","flux_[SII]_6730.82"], rayleigh = False, snr_cut = snr_cut)
        if oi:
            oone = self.get_snr_masked_intensity("flux_[OI]_6300.3", rayleigh = False, snr_cut = snr_cut)

        # calculate masked logs
        log_oiii_hb = np.ma.log10(oiii/hb)
        if nii:
            log_nii_ha = np.ma.log10(ntwo / ha)
        if sii:
            log_sii_ha = np.ma.log10(stwo / ha)
        if oi:
            log_oi_ha = np.ma.log10(oone / ha)

        return {
            "oiii":log_oiii_hb,
            "nii":log_nii_ha if nii else None,
            "sii":log_sii_ha if sii else None,
            "oi":log_oi_ha if oi else None,
        }
    def get_bpt_masks(self, bpt_data = None, nii = True, sii = False, oi = False, snr_cut = 3, use_nii_liner = True):
        """
        Get BPT diagram masks for all categories depending on which diagrams are selected
        Follows code from sdss-marvin
        https://sdss-marvin.readthedocs.io/en/latest/_modules/marvin/utils/dap/bpt.html#bpt_kewley06

        Parameters
        ----------
        nii: 'bool'
            if True, uses [NII] based BPT categories
        bpt_data: 'dict'
            dictionary of BPT data from method get_bpt_masks - will auto load if not provided
        sii: 'bool'
            if True, uses [SII] based BPT categories
        oi: 'bool'
            if True, uses [OI] based BPT categories
        snr_cut: 'float'
            SNR cut to use
        use_nii_liner: 'bool'
            if True, uses Schawinski 2007 LINER/AGN classification line on [NII] BPT diagram
        """
        if bpt_data is None:
            bpt_data = self.get_bpt_data(nii = nii, sii = sii, oi = oi, snr_cut = snr_cut)
        log_oiii_hb = bpt_data["oiii"]
        log_nii_ha = bpt_data["nii"]
        log_sii_ha = bpt_data["sii"]
        log_oi_ha = bpt_data["oi"]

        # Calculates masks for each emission mechanism according to the paper boundaries.
        # The log_nii_ha < 0.05, log_sii_ha < 0.32, etc are necessary because the classification lines
        # diverge and we only want the region before the asymptota.
        if nii:
            sf_mask_nii = ((log_oiii_hb < self.kewley_sf_nii(log_nii_ha)) & (log_nii_ha < 0.05)).filled(False)
        else:
            sf_mask_nii = np.ones_like(log_oiii_hb.data, dtype = bool)
        if sii:
            sf_mask_sii = ((log_oiii_hb < self.kewley_sf_sii(log_sii_ha)) & (log_sii_ha < 0.32)).filled(False)
        else:
            sf_mask_sii = np.ones_like(log_oiii_hb.data, dtype = bool)
        if oi:
            sf_mask_oi = ((log_oiii_hb < self.kewley_sf_oi(log_oi_ha)) & (log_oi_ha < -0.59)).filled(False)
        else:
            sf_mask_oi = np.ones_like(log_oiii_hb.data, dtype = bool)



        sf_mask = sf_mask_nii & sf_mask_sii & sf_mask_oi 

        if nii:
            comp_mask = ((log_oiii_hb > self.kewley_sf_nii(log_nii_ha)) & (log_nii_ha < 0.05)).filled(False) & \
                        ((log_oiii_hb < self.kewley_comp_nii(log_nii_ha)) & (log_nii_ha < 0.465)).filled(False)
        else: 
            comp_mask = np.ones_like(log_oiii_hb.data, dtype = bool)

        comp_mask &= (sf_mask_sii & sf_mask_oi) 

        if nii:
            agn_mask_nii = ((log_oiii_hb > self.kewley_comp_nii(log_nii_ha)) |
                            (log_nii_ha > 0.465)).filled(False)
        else: 
            agn_mask_nii = np.ones_like(log_oiii_hb.data, dtype = bool)

        if sii: 
            agn_mask_sii = ((log_oiii_hb > self.kewley_sf_sii(log_sii_ha)) |
                            (log_sii_ha > 0.32)).filled(False)
        else: 
            agn_mask_sii = np.ones_like(log_oiii_hb.data, dtype = bool)

        if oi:
            agn_mask_oi = ((log_oiii_hb > self.kewley_sf_oi(log_oi_ha)) |
                           (log_oi_ha > -0.59)).filled(False)
        else: 
            agn_mask_oi = np.ones_like(log_oiii_hb.data, dtype = bool)

        agn_mask = agn_mask_nii & agn_mask_sii & agn_mask_oi 

        if sii:
            seyfert_mask_sii = agn_mask & (self.kewley_agn_sii(log_sii_ha) < log_oiii_hb).filled(False)
        else: 
            seyfert_mask_sii = np.ones_like(log_oiii_hb.data, dtype = bool)
        if oi:
            seyfert_mask_oi = agn_mask & (self.kewley_agn_oi(log_oi_ha) < log_oiii_hb).filled(False)
        else: 
            seyfert_mask_oi = np.ones_like(log_oiii_hb.data, dtype = bool)

        seyfert_mask = seyfert_mask_sii & seyfert_mask_oi 

        
        if use_nii_liner & nii:
            liner_mask_nii = agn_mask_nii & (self.schawinski_liner_nii(log_nii_ha) > log_oiii_hb).filled(False)

        if sii:
            liner_mask_sii = agn_mask & (self.kewley_agn_sii(log_sii_ha) > log_oiii_hb).filled(False)
        else: 
            liner_mask_sii = np.ones_like(log_oiii_hb.data, dtype = bool)
        if oi:
            liner_mask_oi = agn_mask & (self.kewley_agn_oi(log_oi_ha) > log_oiii_hb).filled(False)
        else: 
            liner_mask_oi = np.ones_like(log_oiii_hb.data, dtype = bool)

        liner_mask = liner_mask_sii & liner_mask_oi & liner_mask_nii if use_nii_liner else liner_mask_sii & liner_mask_oi

        # The invalid mask is the combination of spaxels that are invalid in all of the emission maps
        if nii:
            invalid_mask_nii = log_oiii_hb.mask | log_nii_ha.mask
        else:
            invalid_mask_nii = log_oiii_hb.mask
        if sii:
            invalid_mask_sii = log_oiii_hb.mask | log_sii_ha.mask
        else:
            invalid_mask_sii = log_oiii_hb.mask
        if oi:
            invalid_mask_oi = log_oiii_hb.mask | log_oi_ha.mask 
        else:
            invalid_mask_oi = log_oiii_hb.mask 

        invalid_mask = invalid_mask_nii | invalid_mask_sii | invalid_mask_oi

        # The ambiguous mask are spaxels that are not invalid but don't fall into any of the
        # emission mechanism classifications.
        ambiguous_mask = ~(sf_mask | comp_mask | seyfert_mask | liner_mask) & ~invalid_mask

        sf_classification = {'global': sf_mask,
                             'nii': sf_mask_nii,
                             'sii': sf_mask_sii}

        comp_classification = {'global': comp_mask,
                               'nii': comp_mask}

        agn_classification = {'global': agn_mask,
                              'nii': agn_mask_nii,
                              'sii': agn_mask_sii}

        seyfert_classification = {'global': seyfert_mask,
                                  'sii': seyfert_mask_sii}

        liner_classification = {'global': liner_mask,
                                'sii': liner_mask_sii}

        if use_nii_liner:
            liner_classification['nii']=liner_mask_nii

        invalid_classification = {'global': invalid_mask,
                                  'nii': invalid_mask_nii,
                                  'sii': invalid_mask_sii}

        ambiguous_classification = {'global': ambiguous_mask}

        if oi:
            sf_classification['oi'] = sf_mask_oi
            agn_classification['oi'] = agn_mask_oi
            seyfert_classification['oi'] = seyfert_mask_oi
            liner_classification['oi'] = liner_mask_oi
            invalid_classification['oi'] = invalid_mask_oi

        return {
                 'sf': sf_classification,
                 'comp': comp_classification,
                 'agn': agn_classification,
                 'seyfert': seyfert_classification,
                 'liner': liner_classification,
                 'invalid': invalid_classification,
                 'ambiguous': ambiguous_classification
                }

    def get_bpt_sf_curve_distance(self, bpt_data = None, nii = True, sii = False, oi = False, snr_cut = 3):
        """
        Gets 2D Distance from Star Forming categorization line from BPT diagrams. 
        Positive represents more AGN/LINER like and Negative is more Star Formation like

        Parameters
        ----------

        bpt_data: 'dict'
            dictionary of BPT data from method get_bpt_masks - will auto load if not provided
        nii: 'bool'
            if True, uses [NII] based BPT categories
        sii: 'bool'
            if True, uses [SII] based BPT categories
        oi: 'bool'
            if True, uses [OI] based BPT categories
        snr_cut: 'float'
            SNR cut to use
        """

        if bpt_data is None:
            bpt_data = self.get_bpt_data(nii = nii, sii = sii, oi = oi, snr_cut = snr_cut)
        log_oiii_hb = bpt_data["oiii"]
        log_nii_ha = bpt_data["nii"]
        log_sii_ha = bpt_data["sii"]
        log_oi_ha = bpt_data["oi"]

        bpt_dist = {
        }
        y_data = log_oiii_hb

        if nii:
            x_data = log_nii_ha

            x_curve = np.linspace(-2.5, 0.1,100)
            y_curve = self.kewley_comp_nii(x_curve)

            # # === Nearest Euclidean distance to curve ===
            curve_points = np.column_stack([x_curve, y_curve])
            data_points  = np.column_stack([x_data, y_data])
            from scipy.spatial.distance import cdist
            dist_matrix = cdist(data_points.astype(np.float16), curve_points.astype(np.float16))

            nearest_idx = np.argmin(dist_matrix, axis=1)
            min_dist = dist_matrix[np.arange(len(x_data)), nearest_idx]

            # nearest curve coordinates
            x_near = x_curve[nearest_idx]
            y_near = y_curve[nearest_idx]

            #==== Signed distance using curve normal===

            # derivative of Kewley curve
            dy_dx = -0.61 / (x_near - 0.47)**2

            #=== normal vector===
            nx = -dy_dx
            ny = np.ones_like(dy_dx)

            # ===vector from curve to point===
            vx = x_data - x_near
            vy = y_data - y_near

            dot = vx * nx + vy * ny

            bpt_dist["nii"] = np.sign(dot) * min_dist

        if sii:
            x_data = log_sii_ha

            x_curve = np.linspace(-2.5, 0.5,100)
            y_curve = self.kewley_sf_sii(x_curve)

            # === Nearest Euclidean distance to curve ===
            curve_points = np.column_stack([x_curve, y_curve])
            data_points  = np.column_stack([x_data, y_data])
            from scipy.spatial.distance import cdist
            dist_matrix = cdist(data_points.astype(np.float16), curve_points.astype(np.float16))

            nearest_idx = np.argmin(dist_matrix, axis=1)
            min_dist = dist_matrix[np.arange(len(x_data)), nearest_idx]

            # nearest curve coordinates
            x_near = x_curve[nearest_idx]
            y_near = y_curve[nearest_idx]

            #==== Signed distance using curve normal===

            # derivative of Kewley curve
            dy_dx = -0.72 / (x_near - 0.32)**2

            #=== normal vector===
            nx = -dy_dx
            ny = np.ones_like(dy_dx)

            # ===vector from curve to point===
            vx = x_data - x_near
            vy = y_data - y_near

            dot = vx * nx + vy * ny

            bpt_dist["sii"] = np.sign(dot) * min_dist

        if oi:
            x_data = log_oi_ha

            x_curve = np.linspace(-3, -1,100)
            y_curve = self.kewley_sf_oi(x_curve)

            # === Nearest Euclidean distance to curve ===
            curve_points = np.column_stack([x_curve, y_curve])
            data_points  = np.column_stack([x_data, y_data])
            from scipy.spatial.distance import cdist
            dist_matrix = cdist(data_points.astype(np.float16), curve_points.astype(np.float16))

            nearest_idx = np.argmin(dist_matrix, axis=1)
            min_dist = dist_matrix[np.arange(len(x_data)), nearest_idx]

            # nearest curve coordinates
            x_near = x_curve[nearest_idx]
            y_near = y_curve[nearest_idx]

            #==== Signed distance using curve normal===

            # derivative of Kewley curve
            dy_dx = -0.73 / (x_near - 0.59)**2

            #=== normal vector===
            nx = -dy_dx
            ny = np.ones_like(dy_dx)

            # ===vector from curve to point===
            vx = x_data - x_near
            vy = y_data - y_near

            dot = vx * nx + vy * ny

            bpt_dist["oi"] = np.sign(dot) * min_dist

        return bpt_dist


    def draw_classification_lines_nii(self, ax, sf_kwargs = {}, comp_kwargs = {}, agn_kwargs = {}, **kwargs):
        """
        Draw classification lines onto nii_ha BPT Diagram

        Parameters
        ----------
        ax: 'matplotlib.pyplot.figure.axes'
            axes to plot lines on
        sf_kwargs: 'dict', optional, must be keyword
            kwargs to pass to ax.plot for the SF Classification Line
        comp_kwargs: 'dict', optional, must be keyword
            kwargs to pass to ax.plot for the COMP Classification Line
        agn_kwargs: 'dict', optional, must be keyword
            kwargs to pass to ax.plot for the AGN/LI(NER Classification Line
        **kwargs: 'dict', optional, must be keyword
            universal line kwargs passed to all
            If kwarg is set to a 3 element array, then they are each passed to 
            SF, Comp, AGN, in that order
        """

        # Check kwargs:

        # Default zorder
        if "zorder" not in kwargs:
            kwargs["zorder"] = 0

        for keyword in kwargs:
            if (kwargs[keyword].__class__ is tuple) | ((kwargs[keyword].__class__ is list)):
                # Should have 3 entries
                if len(kwargs[keyword]) == 3:
                    sf_kwargs[keyword], comp_kwargs[keyword], agn_kwargs[keyword] = kwargs[keyword]
                if len(kwargs[keyword]) == 1:
                    if keyword not in sf_kwargs:
                        sf_kwargs[keyword] = kwargs[keyword]
                    if keyword not in comp_kwargs:
                        comp_kwargs[keyword] = kwargs[keyword]
                    if keyword not in agn_kwargs:
                        agn_kwargs[keyword] = kwargs[keyword]
            else:
                if keyword not in sf_kwargs:
                    sf_kwargs[keyword] = kwargs[keyword]
                if keyword not in comp_kwargs:
                    comp_kwargs[keyword] = kwargs[keyword]
                if keyword not in agn_kwargs:
                    agn_kwargs[keyword] = kwargs[keyword]

        # Default colors:
        if "color" not in sf_kwargs:
            sf_kwargs["color"] = self.pal_colorblind[1]
        if "color" not in comp_kwargs:
            comp_kwargs["color"] = "k"
        if "color" not in agn_kwargs:
            agn_kwargs["color"] = self.pal_colorblind[4]

        # line style
        if ("ls" not in sf_kwargs) & ("linestyle" not in sf_kwargs):
            sf_kwargs["ls"] = "--"

        # line widths
        for kws in [sf_kwargs, comp_kwargs, agn_kwargs]:
            if ("lw" not in kws) & ("linewidth" not in kws):
                kws["lw"] = 2


        # Default Labels:
        if "label" not in sf_kwargs:
            sf_kwargs["label"] = "Kauffmann+03"
        if "label" not in comp_kwargs:
            comp_kwargs["label"] = "Kewley+01"
        if "label" not in agn_kwargs:
            agn_kwargs["label"] = "Schawinski+07"



        # SF Line
        x = np.linspace(-1.5,0.045)
        ax.plot(x, self.kewley_sf_nii(x), **sf_kwargs)

        # Comp Line
        x = np.linspace(-2, 0.4)
        ax.plot(x, self.kewley_comp_nii(x), **comp_kwargs)

        # AGN / LI(N)ER Line
        x = np.linspace(-0.180,1.5)
        ax.plot(x, self.schawinski_liner_nii(x), **agn_kwargs)

        return ax

    def draw_classification_lines_sii(self, ax, sf_kwargs = {}, agn_kwargs = {}, **kwargs):
        """
        Draw classification lines onto sii_ha BPT Diagram

        Parameters
        ----------
        ax: 'matplotlib.pyplot.figure.axes'
            axes to plot lines on
        sf_kwargs: 'dict', optional, must be keyword
            kwargs to pass to ax.plot for the SF Classification Line
        agn_kwargs: 'dict', optional, must be keyword
            kwargs to pass to ax.plot for the AGN/LI(NER Classification Line
        **kwargs: 'dict', optional, must be keyword
            universal line kwargs passed to all
            If kwarg is set to a 3 element array, then they are each passed to 
            SF, Comp, AGN, in that order
        """

        # Check kwargs:

        # Default zorder
        if "zorder" not in kwargs:
            kwargs["zorder"] = 0

        for keyword in kwargs:
            if (kwargs[keyword].__class__ is tuple) | ((kwargs[keyword].__class__ is list)):
                # Should have 3 entries
                if len(kwargs[keyword]) == 2:
                    sf_kwargs[keyword], agn_kwargs[keyword] = kwargs[keyword]
                if len(kwargs[keyword]) == 1:
                    if keyword not in sf_kwargs:
                        sf_kwargs[keyword] = kwargs[keyword]
                    if keyword not in agn_kwargs:
                        agn_kwargs[keyword] = kwargs[keyword]
            else:
                if keyword not in sf_kwargs:
                    sf_kwargs[keyword] = kwargs[keyword]
                if keyword not in agn_kwargs:
                    agn_kwargs[keyword] = kwargs[keyword]

        # Default colors:
        if "color" not in sf_kwargs:
            sf_kwargs["color"] = self.pal_colorblind[1]
        if "color" not in agn_kwargs:
            agn_kwargs["color"] = self.pal_colorblind[4]

        # Default Labels:
        if "label" not in sf_kwargs:
            sf_kwargs["label"] = "Kewley+01"
        if "label" not in agn_kwargs:
            agn_kwargs["label"] = "Kewley+06"



        # SF Line
        x = np.linspace(-2,0.315)
        ax.plot(x, self.kewley_sf_sii(x), **sf_kwargs)

        # AGN Line
        x = np.linspace(-0.308, 1.)
        ax.plot(x, self.kewley_agn_sii(x), **agn_kwargs)

        

        return ax

    def draw_classification_lines_oi(self, ax, sf_kwargs = {}, agn_kwargs = {}, **kwargs):
        """
        Draw classification lines onto oi_ha BPT Diagram

        Parameters
        ----------
        ax: 'matplotlib.pyplot.figure.axes'
            axes to plot lines on
        sf_kwargs: 'dict', optional, must be keyword
            kwargs to pass to ax.plot for the SF Classification Line
        agn_kwargs: 'dict', optional, must be keyword
            kwargs to pass to ax.plot for the AGN/LI(NER Classification Line
        **kwargs: 'dict', optional, must be keyword
            universal line kwargs passed to all
            If kwarg is set to a 3 element array, then they are each passed to 
            SF, Comp, AGN, in that order
        """

        # Check kwargs:

        # Default zorder
        if "zorder" not in kwargs:
            kwargs["zorder"] = 0

        for keyword in kwargs:
            if (kwargs[keyword].__class__ is tuple) | ((kwargs[keyword].__class__ is list)):
                # Should have 3 entries
                if len(kwargs[keyword]) == 2:
                    sf_kwargs[keyword], agn_kwargs[keyword] = kwargs[keyword]
                if len(kwargs[keyword]) == 1:
                    if keyword not in sf_kwargs:
                        sf_kwargs[keyword] = kwargs[keyword]
                    if keyword not in agn_kwargs:
                        agn_kwargs[keyword] = kwargs[keyword]
            else:
                if keyword not in sf_kwargs:
                    sf_kwargs[keyword] = kwargs[keyword]
                if keyword not in agn_kwargs:
                    agn_kwargs[keyword] = kwargs[keyword]

        # Default colors:
        if "color" not in sf_kwargs:
            sf_kwargs["color"] = self.pal_colorblind[1]
        if "color" not in agn_kwargs:
            agn_kwargs["color"] = self.pal_colorblind[4]

        # Default Labels:
        if "label" not in sf_kwargs:
            sf_kwargs["label"] = "Kewley+01"
        if "label" not in agn_kwargs:
            agn_kwargs["label"] = "Kewley+06"



        # SF Line
        x = np.linspace(-3,-0.7)
        ax.plot(x, self.kewley_sf_oi(x), **sf_kwargs)

        # AGN Line
        x = np.linspace(-1.12, 0.5)
        ax.plot(x, self.kewley_agn_oi(x), **agn_kwargs)

        

        return ax

    def plot_bpt_nii(self, bpt_data = None, fig = None, ax = None, 
                     sf_kwargs = {},
                     comp_kwargs = {}, 
                     agn_kwargs = {}, 
                     snr_cut = 3,
                     legend = True,
                     scale_to_dkbpt = False,
                     aspect = None,
                     no_ylabel = False,
                     **kwargs,
                     ):
        """
        bpt_data: 'dict'
            dictionary of BPT data from method get_bpt_masks - will auto load if not provided
        fig: 'matplotlib.pyplot.figure'
            figure to plot on - will get from ax or create if needed
        ax: 'matplotlib.pyplot.figure.axes'
            axes to plot on
        sf_kwargs: 'dict', optional, must be keyword
            kwargs to pass to ax.plot for the SF Classification Line
        comp_kwargs: 'dict', optional, must be keyword
            kwargs to pass to ax.plot for the COMP Classification Line
        agn_kwargs: 'dict', optional, must be keyword
            kwargs to pass to ax.plot for the AGN/LI(NER Classification Line
        snr_cut: 'float'
            SNR cut to use
        legend: 'bool'
            if True, adds legend for classification lines
        scale_to_dkbpt: 'bool'
            if True, scales axes to match that of Krishnarao et al. 2020a
        kwargs: 'dict'
            passed to 'seaborn.displot' for 2D histogramming
        no_ylabel: 'bool'
            skips adding y-axis label if True
        """
        if not hasattr(ax, 'scatter'):
            if not hasattr(fig, 'add_subplot'):
                fig = plt.figure(constrained_layout = True)
                ax = fig.add_subplot(111)
            else:
                ax = fig.add_subplot(111)
        else:
            if not hasattr(fig, 'add_subplot'):
                fig = ax.get_figure()



        if bpt_data is None:
            bpt_data = self.get_bpt_data(nii = True, snr_cut = snr_cut)

        log_oiii_hb = bpt_data["oiii"]
        log_nii_ha = bpt_data["nii"]

        a = ax.hexbin(log_nii_ha, log_oiii_hb, 
                        bins = kwargs.pop("bins","log"), 
                        gridsize = kwargs.pop("gridsize",1000),
                        mincnt = kwargs.pop("mincnt",1),
                        zorder = kwargs.pop("zorder",-1),
                        **kwargs)

        ax = self.draw_classification_lines_nii(ax = ax, 
                                                sf_kwargs = sf_kwargs, 
                                                agn_kwargs = agn_kwargs, 
                                                comp_kwargs=comp_kwargs)


        ax.set_xlabel(r'$log_{10}$([NII] $\lambda 6583$/H$\alpha$)',fontsize=12)
        if not no_ylabel:
            ax.set_ylabel(r'$log_{10}$([OIII] $\lambda 5007$/H$\beta$)',fontsize=12)

        if scale_to_dkbpt:
            ax = self.scale_to_dkbpt(ax)
        else: 
            ax.set_xlim(-2,1)
            ax.set_ylim(-1.5,1.5)
        if legend:
            ax.legend(fontsize = 10)
        if aspect is not None:
            ax.set_aspect(aspect)

        return fig

    def plot_bpt_sii(self, bpt_data = None, fig = None, ax = None, 
                     sf_kwargs = {},
                     agn_kwargs = {}, 
                     snr_cut = 3,
                     legend = True,
                     aspect = None,
                     no_ylabel = False,
                     **kwargs,
                     ):
        """
        bpt_data: 'dict'
            dictionary of BPT data from method get_bpt_masks - will auto load if not provided
        fig: 'matplotlib.pyplot.figure'
            figure to plot on - will get from ax or create if needed
        ax: 'matplotlib.pyplot.figure.axes'
            axes to plot on
        sf_kwargs: 'dict', optional, must be keyword
            kwargs to pass to ax.plot for the SF Classification Line
        comp_kwargs: 'dict', optional, must be keyword
            kwargs to pass to ax.plot for the COMP Classification Line
        agn_kwargs: 'dict', optional, must be keyword
            kwargs to pass to ax.plot for the AGN/LI(NER Classification Line
        snr_cut: 'float'
            SNR cut to use
        legend: 'bool'
            if True, adds legend for classification lines
        kwargs: 'dict'
            passed to 'seaborn.displot' for 2D histogramming
        no_ylabel: 'bool'
            skips adding y-axis label if True
        """
        if not hasattr(ax, 'scatter'):
            if not hasattr(fig, 'add_subplot'):
                fig = plt.figure(constrained_layout = True)
                ax = fig.add_subplot(111)
            else:
                ax = fig.add_subplot(111)
        else:
            if not hasattr(fig, 'add_subplot'):
                fig = ax.get_figure()



        if bpt_data is None:
            bpt_data = self.get_bpt_data(nii = False, sii = True, snr_cut = snr_cut)

        log_oiii_hb = bpt_data["oiii"]
        log_sii_ha = bpt_data["sii"]

        a = ax.hexbin(log_sii_ha, log_oiii_hb, 
                        bins = kwargs.pop("bins","log"), 
                        gridsize = kwargs.pop("gridsize",1000),
                        mincnt = kwargs.pop("mincnt",1),
                        zorder = kwargs.pop("zorder",-1),
                        **kwargs)

        ax = self.draw_classification_lines_sii(ax = ax, 
                                                sf_kwargs = sf_kwargs, 
                                                agn_kwargs = agn_kwargs)


        ax.set_xlabel(r'$log_{10}$([SII] $\lambda 6717+\lambda6730$/H$\alpha$)',fontsize=12)
        if not no_ylabel:
            ax.set_ylabel(r'$log_{10}$([OIII] $\lambda 5007$/H$\beta$)',fontsize=12)

        
        ax.set_xlim(-2,1)
        ax.set_ylim(-1.5,1.5)

        if legend:
            ax.legend(fontsize = 10)
        if aspect is not None:
            ax.set_aspect(aspect)

        return fig

    def plot_bpt_oi(self, bpt_data = None, fig = None, ax = None, 
                     sf_kwargs = {},
                     agn_kwargs = {}, 
                     snr_cut = 3,
                     legend = True,
                     aspect = None,
                     no_ylabel = False,
                     **kwargs,
                     ):
        """
        bpt_data: 'dict'
            dictionary of BPT data from method get_bpt_masks - will auto load if not provided
        fig: 'matplotlib.pyplot.figure'
            figure to plot on - will get from ax or create if needed
        ax: 'matplotlib.pyplot.figure.axes'
            axes to plot on
        sf_kwargs: 'dict', optional, must be keyword
            kwargs to pass to ax.plot for the SF Classification Line
        comp_kwargs: 'dict', optional, must be keyword
            kwargs to pass to ax.plot for the COMP Classification Line
        agn_kwargs: 'dict', optional, must be keyword
            kwargs to pass to ax.plot for the AGN/LI(NER Classification Line
        snr_cut: 'float'
            SNR cut to use
        legend: 'bool'
            if True, adds legend for classification lines
        no_ylabel: 'bool'
            skips adding y-axis label if True
        kwargs: 'dict'
            passed to 'seaborn.displot' for 2D histogramming
        """
        if not hasattr(ax, 'scatter'):
            if not hasattr(fig, 'add_subplot'):
                fig = plt.figure(constrained_layout = True)
                ax = fig.add_subplot(111)
            else:
                ax = fig.add_subplot(111)
        else:
            if not hasattr(fig, 'add_subplot'):
                fig = ax.get_figure()



        if bpt_data is None:
            bpt_data = self.get_bpt_data(nii = False, oi = True, snr_cut = snr_cut)

        log_oiii_hb = bpt_data["oiii"]
        log_oi_ha = bpt_data["oi"]

        a = ax.hexbin(log_oi_ha, log_oiii_hb, 
                        bins = kwargs.pop("bins","log"), 
                        gridsize = kwargs.pop("gridsize",1000),
                        mincnt = kwargs.pop("mincnt",1),
                        zorder = kwargs.pop("zorder",-1),
                        **kwargs)

        ax = self.draw_classification_lines_oi(ax = ax, 
                                                sf_kwargs = sf_kwargs, 
                                                agn_kwargs = agn_kwargs)


        ax.set_xlabel(r'$log_{10}$([OI] $\lambda 6300$/H$\alpha$)',fontsize=12)
        if not no_ylabel:
            ax.set_ylabel(r'$log_{10}$([OIII] $\lambda 5007$/H$\beta$)',fontsize=12)

        
        ax.set_xlim(-3,1)
        ax.set_ylim(-1.5,1.5)

        if legend:
            ax.legend(fontsize = 10)
        if aspect is not None:
            ax.set_aspect(aspect)

        return fig

    def plot_multi_bpt(self, bpt_data = None, fig = None, axes = None,
                       nii = True, 
                       sii = True, 
                       oi = True, 
                       sf_kwargs = {},
                       comp_kwargs = {},
                       agn_kwargs = {}, 
                       snr_cut = 3,
                       legend = False,
                       aspect = None,
                       **kwargs, ):
        """
        bpt_data: 'dict'
            dictionary of BPT data from method get_bpt_masks - will auto load if not provided
        fig: 'matplotlib.pyplot.figure'
            figure to plot on - will get from ax or create if needed
        axes: 'dict' of 'matplotlib.pyplot.figure.axes'
            axes to plot on in order of nii, sii, oi (or fewer if fewer are specified)
            must have entires with keys ["nii", "sii", "oi"] or subset
        sf_kwargs: 'dict', optional, must be keyword
            kwargs to pass to ax.plot for the SF Classification Line
        comp_kwargs: 'dict', optional, must be keyword
            kwargs to pass to ax.plot for the COMP Classification Line
        agn_kwargs: 'dict', optional, must be keyword
            kwargs to pass to ax.plot for the AGN/LI(NER Classification Line
        snr_cut: 'float'
            SNR cut to use
        legend: 'bool'
            if True, adds legend for classification lines
        kwargs: 'dict'
            passed to 'seaborn.displot' for 2D histogramming
        """
        N_bpt = np.sum([nii, sii, oi])
        if axes is not None:
            assert N_bpt == len(axes)

            if fig is None:
                fig = axes["nii" if nii else "sii" if sii else "oi"].get_figure()
        else:
            if fig is None:
                layout = ["nii" if nii else None, "sii" if sii else None, "oi" if oi else None]
                layout = list(filter(lambda x: x is not None, layout))
                fig,axes = plt.subplot_mosaic([layout], 
                                              figsize = kwargs.pop("figsize", (9,3)),
                                              constrained_layout = True, 
                                              sharey = True)
            else:
                axes = fig.subplot_mosaic([layout], sharey = True)


        if bpt_data is None:
            bpt_data = self.get_bpt_data(nii = nii, sii = sii, oi = oi, snr_cut = snr_cut)

        if nii:
            fig = self.plot_bpt_nii(bpt_data = bpt_data, fig = fig, ax = axes["nii"], 
                              sf_kwargs = sf_kwargs, 
                              comp_kwargs = comp_kwargs,
                              agn_kwargs = agn_kwargs,
                              legend = legend, 
                              aspect = aspect, 
                              **kwargs)

        if sii:
            fig = self.plot_bpt_sii(bpt_data = bpt_data, fig = fig, ax = axes["sii"], 
                              sf_kwargs = sf_kwargs, 
                              agn_kwargs = agn_kwargs,
                              legend = legend, 
                              aspect = aspect, 
                              no_ylabel = True if nii else False,
                              **kwargs)

        if oi:
            fig = self.plot_bpt_oi(bpt_data = bpt_data, fig = fig, ax = axes["oi"], 
                              sf_kwargs = sf_kwargs, 
                              agn_kwargs = agn_kwargs,
                              legend = legend, 
                              aspect = aspect, 
                              no_ylabel = True if nii|sii else False,
                              **kwargs)

        return fig












        
        
    def scale_to_dkbpt(self, ax):
        """
        Scales x and y axis to match Krishnarao+19 BPT Diagram

        Parameters
        ----------
        ax 'matplotlib.pyplot.figure.axes'
            axes to plot lines on
        """
        Xmin, Xmax         = -1.2, 1.0
        Ymin, Ymax         = -1.2, 1.0
        ax.set_xlim([Xmin, Xmax])
        ax.set_ylim([Ymin, Ymax])

        ax.set_aspect("equal")
        return ax



    def sky_section(self, bounds, radius = None, wrap_at_180 = True):
        """
        Extract a sub section of the survey from the sky

        Parameters
        ----------

        bounds: `list` or `Quantity` or `SkyCoord`
            if `list` or `Quantity` must be formatted as:
                [min Galactic Longitude, max Galactic Longitude, min Galactic Latitude, max Galactic Latitude]
                or 
                [center Galactic Longitude, center Galactic Latitude] and requires radius keyword to be set
                default units of u.deg are assumed
            if `SkyCoord', must be length 4 or length 1 or length 2
                length 4 specifies 4 corners of rectangular shape
                length 1 specifies center of circular region and requires radius keyword to be set
                length 2 specifies two corners of rectangular region
        radius: 'number' or 'Quantity', optional, must be keyword
            sets radius of circular region
        wrap_at_180: `bool`, optional, must be keyword
            if True, wraps longitude angles at 180d
            use if mapping accross Galactic Center
        """
        if wrap_at_180:
            wrap_at = "180d"
        else:
            wrap_at = "360d"

        if not isinstance(bounds, u.Quantity) | isinstance(bounds, SkyCoord):
            bounds *= u.deg
            logging.warning("No units provided for bounds, assuming u.deg")

        lvm_coords = self.get_SkyCoord()

        if isinstance(bounds, SkyCoord):
            if len(bounds) == 1:
                if radius is None:
                    raise TypeError("Radius must be provided if only a single coordinate is given")
                elif not isinstance(radius, u.Quantity):
                    radius *= u.deg
                    logging.warning("No units provided for radius, assuming u.deg")
                center = bounds
            elif len(bounds) >= 2:
                min_lon, max_lon = bounds.l.wrap_at(wrap_at).min(), bounds.l.wrap_at(wrap_at).max()
                min_lat, max_lat = bounds.b.min(), bounds.l.max()
                if min_lon == max_lon:
                    if wrap_at == "180d":
                        min_lon, max_lon = -180*u.deg, 179.999999999*u.deg
                    else:
                        min_lon, max_lon = 0*u.deg, 359.99999999*u.deg
        elif len(bounds) == 2:
            if radius is None:
                raise TypeError("Radius must be provided if only a single coordinate is given")
            elif not isinstance(radius, u.Quantity):
                radius *= u.deg
                logging.warning("No units provided for radius, assuming u.deg")
            center = SkyCoord(l = bounds[0], b = bounds[1], frame = 'galactic')
        elif len(bounds) == 4:
            min_lon, max_lon, min_lat, max_lat = Angle(bounds)
            min_lon = min_lon.wrap_at(wrap_at)
            max_lon = max_lon.wrap_at(wrap_at)
            if min_lon == max_lon:
                if wrap_at == "180d":
                    min_lon, max_lon = -180*u.deg, 179.999999999*u.deg
                else:
                    min_lon, max_lon = 0*u.deg, 359.99999999*u.deg
        else:
            raise TypeError("Input bounds and/or radius are not understood")

        # rectangular extraction
        if radius is None:
            # Mask of points inside rectangular region
            inside_mask = lvm_coords.l.wrap_at(wrap_at) <= max_lon
            inside_mask &= lvm_coords.l.wrap_at(wrap_at) >= min_lon
            inside_mask &= lvm_coords.b <= max_lat
            inside_mask &= lvm_coords.b >= min_lat

        else: # Circle extraction
            # Compute Separation
            # Warning to self: This is VERY slow
            sep = lvm_coords.separation(center)

            # Mask of points inside circular region
            inside_mask = sep <= radius

        return self[inside_mask]

    def intensity_map(
        self,
        c,
        fig = None, 
        ax = None,
        lrange=None,
        brange=None,
        bounds = None,
        radius = None,
        wrap_at_180 = True,
        power=0.3,
        norm = None,
        rayleigh = True,
        snr_cut = 3,
        percentile_clip = None,
        verbose = False,
        colorbar = True,
        cbar_kwargs = {},
        s_factor = 1.0,
        aspect = None,
        smooth = False,
        smooth_res = None,
        **kwargs,
        ):
        """
        Plot a galactic map of flux from a single emission line using a colormap.

        Parameters
        ----------
            c:'str', 'list-like'
                name of column to plot map of if a string
                if list-like, will pass as keyword to scatter plot
            fig: 'plt.figure', optional, must be keyword
                if provided, will create axes on the figure provided
            ax: 'plt.figure.axes' or 'cartopy.axes`, optional, must be keyword
                if provided, will plot on these axes
                can provide ax as a cartpy projection that contains different map projections
            lrange (tuple): Optional (min, max) Galactic longitude range.
            brange (tuple): Optional (min, max) Galactic latitude range.
            bounds: `list` or `Quantity` or `SkyCoord`
                if provided, will ignore l_range and b_range
                if `list` or `Quantity` must be formatted as:
                    [min Galactic Longitude, max Galactic Longitude, min Galactic Latitude, max Galactic Latitude]
                    or 
                    [center Galactic Longitude, center Galactic Latitude] and requires radius keyword to be set
                    default units of u.deg are assumed
                if `SkyCoord', must be length 4 or length 1 or length 2
                    length 4 specifies 4 corners of rectangular shape
                    length 1 specifies center of circular region and requires radius keyword to be set
                    length 2 specifies two corners of rectangular region
            radius: 'number' or 'Quantity', optional, must be keyword
                sets radius of circular region
                only used if bounds is provided and is 2d
            wrap_at_180: `bool`, optional, must be keyword
                if True, wraps longitude angles at 180d
                use if mapping accross Galactic Center
            norm: 'matplotlib.colors.Normalize' like
                defaults to LogNorm if none provided
            rayleigh: 'bool'
                if True, plots flux in units of Rayleighs instead of default energy flux
            snr_cut: 'float'
                min SNR to use
            percentile_clip: list-like
                [low,high] percentile amounts to clip data within
            colorbar: 'bool', optional, must be keyword
                if True, plots colorbar
            cbar_kwargs: 'dict', optional, must be keyword
                dictionary of kwargs to pass to colorbar
            s_factor: 'number', optional, must be keyword
                multiplied by supplied default s value to set size of lvm beams
            aspect: 'str','float'
                if provided, passed to ax.set_aspect(aspect)
            smooth: `bool`, optional, must be keyword
                if True, smooths map using griddata and plots using pcolormesh
            smooth_res: `number`, optional, must be keyword
                pixel width in units of arcsec


        """
        if not hasattr(ax, 'scatter'):
            if not hasattr(fig, 'add_subplot'):
                fig = plt.figure(constrained_layout = True)
                ax = fig.add_subplot(111)
            else:
                ax = fig.add_subplot(111)
        else:
            if not hasattr(fig, 'add_subplot'):
                fig = ax.get_figure()

        # Filter by galactic coordinate ranges
        if bounds is None:
            if (lrange is None) & (brange is None):
                tab_filtered = self
            else:
                if lrange is None:
                    lrange = [-180,180]*u.deg
                elif brange is None:
                    brange = [-90,90]*u.deg

                bounds = [*lrange, *brange]

                tab_filtered = self.sky_section(bounds = bounds, radius = radius, wrap_at_180 = wrap_at_180)
        else:
            tab_filtered = self.sky_section(bounds = bounds, radius = radius, wrap_at_180 = wrap_at_180)

        if c.__class__ is str:
            colname = c
            flux = tab_filtered.get_snr_masked_intensity(colname, snr_cut = snr_cut, rayleigh = rayleigh)

            if (percentile_clip is not None) & (norm is None):
                if len(percentile_clip) != 2:
                    logging.warning("percentile_clip must only have two values in tuple or list form - ignoring!")
                else:
                    low,high = np.percentile(flux, percentile_clip)
                    norm = LogNorm(vmin = low, vmax = high)


        elif norm is None:
            norm = LogNorm()

        if "cmap" not in kwargs:
            kwargs["cmap"] = "inferno"



        if hasattr(ax, "coastlines"):
            if not "transform" in kwargs:
                kwargs["transform"] = ccrs.PlateCarree()
                logging.warning("No transform specified with cartopy axes projection, assuming PlateCarree")

        lon_points = Angle(tab_filtered["GAL-LON"]).wrap_at("180d")
        lat_points = tab_filtered["GAL-LAT"]

        if lrange is None:
            lrange_s = [lon_points.max().value, lon_points.min().value]
        elif isinstance(lrange, u.Quantity):
            lrange = Angle(lrange).wrap_at(wrap_at).value
        else:
            logging.warning("No units provided for lrange, assuming u.deg")
            lrange = Angle(lrange*u.deg).wrap_at(wrap_at).value
        if brange is None:
            brange_s = [lat_points.min(), lat_points.max()]
        elif isinstance(brange, u.Quantity):
            brange = brange.to(u.deg).value

        # Compute size of points (attempt)
        if not smooth:
            if not "s" in kwargs:
                size = fig.get_size_inches()*fig.dpi
                if brange is not None:
                    brange_s = brange
                else:
                    brange_s = [tab_filtered["GAL-LAT"].min(), tab_filtered["GAL-LAT"].max()]

                if lrange is not None:
                    lrange_s = lrange
                else:
                    lrange_s = [tab_filtered["GAL-LON"].min(), tab_filtered["GAL-LON"].max()]
                fiber_size_factor = 35.3*u.arcsec / u.deg 
                fiber_size_factor = fiber_size_factor.decompose()
                s = np.min(np.array([size / np.abs(np.diff(lrange_s)), 
                                    size / np.abs(np.diff(brange_s))])) * s_factor

                s*= fiber_size_factor
            
                kwargs["s"] = s

        if smooth:
            if smooth_res is None:
                smooth_res = 30.*u.arcsec #arcsec
            if isinstance(smooth_res, u.Quantity):
                smooth_res = smooth_res.to(u.deg).value
            else:
                smooth_res *= u.arcsec
                smooth_res = smooth_res.to(u.deg).value

            if lrange is None:
                lrange = lrange_s
            if brange is None:
                brange = brange_s
            if lrange[1] < lrange[0]:
                gridx = np.flip(np.arange(lrange[1], lrange[0] + smooth_res, smooth_res))
            else:
                gridx = np.arange(lrange[0], lrange[1] + smooth_res, smooth_res)
            gridy = np.arange(brange[0], brange[1] + smooth_res, smooth_res)

            

            if c.__class__ is str:
                zi = griddata((lon_points, lat_points), flux.value,
                    (gridx[None,:], gridy[:,None]), 
                    method='cubic')
            else:
                zi = griddata((lon_points, lat_points), c,
                    (gridx[None,:], gridy[:,None]), 
                    method='cubic')

        if hasattr(ax, "wcs"):
            if ax.wcs.naxis == 3:
                if smooth:
                    gridx, gridy, _ = ax.wcs.wcs_world2pix(gridx, gridy, np.zeros_like(gridx), 0)
                else:
                    lon_points, lat_points, _ = ax.wcs.wcs_world2pix(lon_points, lat_points, np.zeros_like(lon_points.value), 0)
            elif ax.wcs.naxis == 2:
                if smooth:
                    gridx, gridy = ax.wcs.wcs_world2pix(gridx, gridy, 0)
                else:
                    lon_points, lat_points = ax.wcs.wcs_world2pix(lon_points, lat_points, 0)

        if smooth:
            if verbose:
                print("Plotting smooth map of region...")
                # print(kwargs["cmap"])
            sc = ax.pcolormesh(gridx, gridy, zi, norm = norm, 
                                **kwargs)
            ax.set_xlabel("Galactic Longitude (deg)")
            ax.set_ylabel("Galactic Latitude (deg)")
            # if c.__class__ is str:
                # ax.set_title("{0} ({1})".format(colname, flux.data.unit))



        else:       

            if verbose:
                print("Plotting {} points...".format(len(tab_filtered)))
            # Plot
            if c.__class__ is str:
                sc = ax.scatter(lon_points, 
                           lat_points, 
                           c = flux.value,                 
                           edgecolors='none', 
                           norm = norm,
                           **kwargs)
                ax.set_xlabel("Galactic Longitude (deg)")
                ax.set_ylabel("Galactic Latitude (deg)")
                # ax.set_title("{0} ({1})".format(colname, flux.data.unit))
            else:
                sc = ax.scatter(lon_points, 
                           lat_points, 
                           c = c,                  
                           edgecolors='none', 
                           norm = norm,
                           **kwargs)
                ax.set_xlabel("Galactic Longitude (deg)")
                ax.set_ylabel("Galactic Latitude (deg)")
                # ax.set_title("{0} ({1})".format())
        ax.grid(alpha=0.3)
        if aspect is not None:
            ax.set_aspect(aspect)

        if not hasattr(ax, "coastlines"):
            if lrange is not None:
                ax.set_xlim(lrange)
            else:
                ax.invert_xaxis()

            if brange is not None:
                ax.set_ylim(brange)
            
            ax.set_xlabel("Galactic Longitude (deg)", fontsize = 12)
            ax.set_ylabel("Galactic Latitude (deg)", fontsize = 12)
        else:
            
            if (lrange is not None) & (brange is not None):
                ax.set_extent([lrange[0], lrange[1], brange[0], brange[1]])   
            ax.invert_xaxis()
            try:
                ax.gridlines(draw_labels = True)
            except TypeError:
                ax.gridlines()

        if colorbar:
            if not "label" in cbar_kwargs:
                if c.__class__ is str:
                    cbar_kwargs["label"] = "{} ({})".format(colname, flux.unit)

                else:
                    cbar_kwargs["label"] = None
            cb = plt.colorbar(sc, **cbar_kwargs)


        return fig






    def plot_line_ratios_sii_nii_halpha(self,
                                            seaborn = True,
                                            snr_cut = 3,
                                            kind='hex',
                                            gridsize=50,
                                            bins = "log",
                                            scaling_factor=None,
                                            min_abs_b_deg=0,
                                            max_abs_b_deg=90,
                                            stretch='linear',
                                            power=0.3,
                                            save_dir=None,
                                            show_plots=False,
                                            histlog = True,
                                            max_xval = 1.2,
                                            max_yval = 1.2,
                                            clip = False,
                                            clip_value = 99.9,
                                            max_halpha = None,
                                            **kwargs
                                            ):
        """
        Plot [S II]/Halpha vs [N II]/Halpha with model lines for temperature and sulfur ionization

        Parameters
        ----------
        seaborn: 'bool'
            if True, uses seabons jointplot
        snr_cut: 'float'
            Signal to Noise cut to impose on data - default of 3
        kind: 'str',
            kind of plot to make ["scatter", "hex", etc]
        gridsize: 'int'
            number of bins to use for histogramming
        bins: 'str'
            scaling to use for colormapping - default of log

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
        clip: 'bool'
            if True, clips outlier values using a percentile cut from clip_value
        clip_value: 'float'
            if clip is True, will use this value as the percentile cut
        max_halpha: 'u.Quantity', float,
            max H alpha flux to consider - must be a u.Quantity (lvm flux units or Rayleighs)
            if float - assumes Rayleighs
        """
        # set up
        line_prefix='flux_'
        wave_halpha=6563.0
        wave_nii=6584.0
        wave_sii=6716.0
        ra_col='ra'
        dec_col='dec'
        
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
        print(wave_nii)
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
        # h = 6.626e-27  # erg·s
        # c = 2.998e18   # Å/s
        # rayleigh_factor = 1e6 / (4 * np.pi)

        # def to_rayleighs(flux, wave):
        #   photons = (flux * wave) / (h * c)
        #   return photons / rayleigh_factor

        # wave_halpha_actual = float(re.search(r'_(\d+\.\d+)$', col_halpha).group(1))
        # wave_nii_actual = float(re.search(r'_(\d+\.\d+)$', col_nii).group(1))
        # wave_sii_actual = float(re.search(r'_(\d+\.\d+)$', col_sii).group(1))
        # # wave_sii_6731_actual = float(re.search(r'_(\d+\.\d+)$', col_sii_6731).group(1))

        # flux_halpha = np.asarray(self[col_halpha], dtype=float)
        # flux_nii = np.asarray(self[col_nii], dtype=float)
        # flux_sii = np.asarray(self[col_sii], dtype=float)
        # flux_sii_6731 = np.asarray(self[col_sii_6731], dtype=float)

        


        halpha = self.convert_flux_to_rayleigh(col_halpha).value
        nii = self.convert_flux_to_rayleigh(col_nii).value
        sii = self.convert_flux_to_rayleigh(col_sii).value

        if max_halpha is not None:
            if max_halpha.__class__ is u.Quantity:
                if max_halpha.unit is u.lvm_flux:
                    ha_mask = self[col_halpha] > max_halpha
                elif max_halpha.unit is u.rayleigh:
                    ha_mask = halpha > max_halpha.value
            else:
                ha_mask = halpha > max_halpha

        snr_halpha = self[col_halpha]/self["e_{}".format(col_halpha)]
        snr_nii = self[col_nii]/self["e_{}".format(col_nii)]
        snr_sii = self[col_sii]/self["e_{}".format(col_sii)]

        if max_halpha is not None:
            ray_halpha = np.ma.array(halpha, mask = (snr_halpha < snr_cut)|(~lat_mask)|(ha_mask))
            ray_nii = np.ma.array(nii, mask = (snr_nii < snr_cut)|(~lat_mask)|(ha_mask))
            ray_sii = np.ma.array(sii, mask = (snr_sii < snr_cut)|(~lat_mask)|(ha_mask))
        else:
            ray_halpha = np.ma.array(halpha, mask = (snr_halpha < snr_cut)|(~lat_mask))
            ray_nii = np.ma.array(nii, mask = (snr_nii < snr_cut)|(~lat_mask))
            ray_sii = np.ma.array(sii, mask = (snr_sii < snr_cut)|(~lat_mask))

        

        if scaling_factor:
            ray_halpha *= scaling_factor
            ray_nii *= scaling_factor
            ray_sii *= scaling_factor

        # --- Compute ratios ---
        nii_halpha = ray_nii / ray_halpha
        sii_halpha = ray_sii / ray_halpha

        # --- Apply latitude mask and clean ---
        df = pd.DataFrame({
            'nii_halpha': nii_halpha,
            'sii_halpha': sii_halpha
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

        if clip:
            # --- Clip outliers ---
            for col in df.columns:
                high = np.percentile(df[col], clip_value)
                df = df[df[col] <= high]

        print(f"📈 Plotting {len(df)} points")


        if seaborn:
            fig = sns.jointplot(data = df, x = "nii_halpha", y = "sii_halpha", 
                                kind = kind, 
                                joint_kws = {'bins':bins}, 
                                color = kwargs.pop("color","#4cb391"),
                                **kwargs)

            fig.set_axis_labels(f'[N II]/Hα{label_suffix}',
                                f'[S II]/Hα{label_suffix}', 
                                fontsize = 12)
            fig.fig.suptitle(f'|b| ∈ [{min_abs_b_deg}°, {max_abs_b_deg}°]', 
                            fontsize = 12)

        else:
            # --- Plotting Layout ---
            fig = plt.figure(figsize=(8, 8))
            gs = fig.add_gridspec(4, 4)
            ax_main = fig.add_subplot(gs[:3, :3])
            ax_xhist = fig.add_subplot(gs[3, :3], sharex = ax_main)
            ax_yhist = fig.add_subplot(gs[:3, 3], sharey = ax_main)

            if kind == 'hex':
                cmap = kwargs.pop('cmap', 'plasma')
                hb = ax_main.hexbin(
                    df['nii_halpha'], df['sii_halpha'],
                    gridsize=gridsize,
                    cmap=cmap,
                    mincnt=1,
                    bins=bins,
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
            ax_main.text(0.9, 0.75, f'{0.75}', color='b', verticalalignment='center', rotation=45, fontsize = 12)

            # --- Axis Labels & Histograms ---
            ax_main.set_xlabel(f'[N II]/Hα{label_suffix}', fontsize = 12)
            ax_main.set_xlim(0,max_xval)
            ax_main.set_ylabel(f'[S II]/Hα{label_suffix}', fontsize = 12)
            ax_main.set_ylim(0,max_yval)
            ax_main.set_title(f'|b| ∈ [{min_abs_b_deg}°, {max_abs_b_deg}°]')
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

    def plot_line_ratios_vs_halpha_stacked(self,
                                            snr_cut = 3,
                                            kind='hex',
                                            gridsize=50,
                                            bins = "log",
                                            scaling_factor=None,
                                            min_abs_b_deg=0,
                                            max_abs_b_deg=90,
                                            stretch='linear',
                                            power=0.5,
                                            save_dir=None,
                                            clip = False, 
                                            clip_value = 99.9,
                                            max_halpha = None,
                                            **kwargs):

        line_prefix='flux_'
        wave_halpha=6563.0
        wave_nii=6584.0
        wave_sii=6716.0
        ra_col='ra'
        dec_col='dec'
    
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

        print("🔍 Matched Emission Lines:")
        print(f"  Hα:        {col_halpha}")
        print(f"  [N II]:    {col_nii}")
        print(f"  [S II] 6716: {col_sii}")

        # --- Galactic Latitude Filter ---
        coords = SkyCoord(ra=np.asarray(self[ra_col], dtype=float) * u.deg,
                          dec=np.asarray(self[dec_col], dtype=float) * u.deg,
                          frame='icrs')
        b_deg = coords.galactic.b.deg
        lat_mask = (np.abs(b_deg) >= min_abs_b_deg) & (np.abs(b_deg) <= max_abs_b_deg)
        print(f"📌 {np.sum(lat_mask)} sources within |b| ∈ [{min_abs_b_deg}, {max_abs_b_deg}]°")

        # --- Extract fluxes and convert to Rayleighs ---
        # h = 6.626e-27  # erg·s
        # c = 2.998e18   # Å/s
        # rayleigh_factor = 1e6 / (4 * np.pi)

        # def to_rayleighs(flux, wave):
        #   photons = (flux * wave) / (h * c)
        #   return photons / rayleigh_factor

        # def extract_wave(colname):
        #   return float(re.search(r'_(\d+\.\d+)$', colname).group(1))

        # wave_halpha_actual = extract_wave(col_halpha)
        # wave_nii_actual = extract_wave(col_nii)
        # wave_sii_actual = extract_wave(col_sii)

        # flux_halpha = np.asarray(self[col_halpha], dtype=float)
        # flux_nii = np.asarray(self[col_nii], dtype=float)
        # flux_sii = np.asarray(self[col_sii], dtype=float)

        halpha = self.convert_flux_to_rayleigh(col_halpha).value
        nii = self.convert_flux_to_rayleigh(col_nii).value
        sii = self.convert_flux_to_rayleigh(col_sii).value

        if max_halpha is not None:
            if max_halpha.__class__ is u.Quantity:
                if max_halpha.unit is lvm_flux_unit:
                    ha_mask = self[col_halpha] > max_halpha
                elif max_halpha.unit is u.rayleigh:
                    ha_mask = halpha > max_halpha.value
                else:
                    ha_mask = halpha > max_halpha.to(u.rayleigh).value
            else:
                ha_mask = halpha > max_halpha

        snr_halpha = self[col_halpha]/self["e_{}".format(col_halpha)]
        snr_nii = self[col_nii]/self["e_{}".format(col_nii)]
        snr_sii = self[col_sii]/self["e_{}".format(col_sii)]

        if max_halpha is not None:
            ray_halpha = np.ma.array(halpha, mask = (snr_halpha < snr_cut)|(~lat_mask)|(ha_mask))
            ray_nii = np.ma.array(nii, mask = (snr_nii < snr_cut)|(~lat_mask)|(ha_mask))
            ray_sii = np.ma.array(sii, mask = (snr_sii < snr_cut)|(~lat_mask)|(ha_mask))
        else:
            ray_halpha = np.ma.array(halpha, mask = (snr_halpha < snr_cut)|(~lat_mask))
            ray_nii = np.ma.array(nii, mask = (snr_nii < snr_cut)|(~lat_mask))
            ray_sii = np.ma.array(sii, mask = (snr_sii < snr_cut)|(~lat_mask))

        # ray_halpha = to_rayleighs(flux_halpha, wave_halpha_actual)
        # ray_nii = to_rayleighs(flux_nii, wave_nii_actual)
        # ray_sii = to_rayleighs(flux_sii, wave_sii)

        if scaling_factor:
            ray_halpha *= scaling_factor
            ray_nii *= scaling_factor
            ray_sii *= scaling_factor

        # --- Compute ratios ---
        nii_halpha = ray_nii / ray_halpha
        sii_halpha = ray_sii / ray_halpha

        # --- Apply mask and clean ---
        df = pd.DataFrame({
            'halpha': ray_halpha,
            'nii_halpha': nii_halpha,
            'sii_halpha': sii_halpha
        }).replace([np.inf, -np.inf], np.nan).dropna()

        df = df[(df > 0).all(axis=1)]
        if df.empty:
            print("⚠️ No valid data after filtering.")
            return


        # --- Stretch ---
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

        if clip:
            # --- Clip outliers ---
            for col in ['halpha']:
                high = np.percentile(df[col], clip_value)
                df = df[df[col] <= high]

        print(f"📈 Plotting {len(df)} points")

        # --- Stacked Plot Layout ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True, constrained_layout=True)

        if kind == 'hex':
            cmap1 = kwargs.pop('cmap1', 'viridis')
            cmap2 = kwargs.pop('cmap2', 'plasma')

            hb1 = ax1.hexbin(
                df['halpha'], df['nii_halpha'],
                gridsize=gridsize, cmap=cmap1, bins=bins, **kwargs
            )
            fig.colorbar(hb1, ax=ax1, label='log10(Counts)')

            hb2 = ax2.hexbin(
                df['halpha'], df['sii_halpha'],
                gridsize=gridsize, cmap=cmap2, bins=bins, **kwargs
            )
            fig.colorbar(hb2, ax=ax2, label='log10(Counts)')

        elif kind == 'scatter':
            ax1.scatter(df['halpha'], df['nii_halpha'], s=5, alpha=0.3, color='blue')
            ax2.scatter(df['halpha'], df['sii_halpha'], s=5, alpha=0.3, color='red')

        # --- Labels ---
        ax1.set_ylabel('[N II]/Hα' + label_suffix, fontsize = 12)
        ax2.set_ylabel('[S II]/Hα' + label_suffix, fontsize = 12)
        ax2.set_xlabel('Hα Flux (Rayleighs)' + label_suffix, fontsize = 12)

        ax1.set_title(f'Emission Line Ratios vs Hα\n|b| ∈ [{min_abs_b_deg}°, {max_abs_b_deg}°]', fontsize = 12)
        ax1.grid(alpha=0.3)
        ax2.grid(alpha=0.3)




        # --- Save ---
        if save_dir is not None:
            fname = f'stacked_sii_nii_vs_halpha_{kind}_{stretch}_[R]_ [{min_abs_b_deg}°, {max_abs_b_deg}°].png'
            path = os.path.join(save_dir, fname)
            fig.savefig(path)
            print(f"✅ Saved: {path}")

        return fig


    def plot_lv(self,
                colname,
                vel_diff_colname = None,
                fig = None,
                ax = None,
                lrange = None,
                brange = None,
                bounds = None,
                radius = None,
                snr_cut = 5,
                kind='hex',
                gridsize=50,
                bins = "log",
                rayleigh = True,
                velocity_bounds = None,
                wrap_at_180 = True,
                longitude_colname = "GAL-LON",
                colorbar = True,
                clip_flux = None,
                label_axes = True,
                sigma = 2,
                auto_level_factors = None,
                n_levels = 6,
                cbar_kwargs = {},
                **kwargs
                ):
        """
        Plot longitude vs. velocity diagram for specified flux column

        Parameters
        ----------
        colname: 'str'
            column name for flux to use for plot
        vel_diff_colname: 'str', optional, must be keyword
            if provided will colormap the lv diagram by intensity weighted velocity difference
            intensity weighting is done with colname
            velocity difference is from colname - vel_diff_colname
        fig: 'plt.figure', optional, must be keyword
            if provided, will create axes on the figure provided
        ax: 'plt.figure.axes' or 'cartopy.axes`, optional, must be keyword
            if provided, will plot on these axes
            can provide ax as a cartpy projection that contains different map projections
        lrange (tuple): Optional 
            (min, max) Galactic longitude range.
        brange (tuple): Optional 
            (min, max) Galactic latitude range.
        bounds: `list` or `Quantity` or `SkyCoord`
            if provided, will ignore l_range and b_range
            if `list` or `Quantity` must be formatted as:
                [min Galactic Longitude, max Galactic Longitude, min Galactic Latitude, max Galactic Latitude]
                or 
                [center Galactic Longitude, center Galactic Latitude] and requires radius keyword to be set
                default units of u.deg are assumed
            if `SkyCoord', must be length 4 or length 1 or length 2
                length 4 specifies 4 corners of rectangular shape
                length 1 specifies center of circular region and requires radius keyword to be set
                length 2 specifies two corners of rectangular region
        radius: 'number' or 'Quantity', optional, must be keyword
            sets radius of circular region
            only used if bounds is provided and is 2d
        snr_cut: 'float'
            Signal to Noise cut to impose on data - default of 3
        kind: 'str',
            kind of plot to make ["scatter", "hex", etc]
        gridsize: 'int'
            number of bins to use for histogramming
        bins: 'str'
            scaling to use for colormapping - default of log
        rayleigh: 'bool'
            if True, converts flux to rayleighs
        velocity_bounds: 'list-like'
            [min,max] velocity to use for velocity axis
        wrap_at_180: 'bool'
            if True, wraps coordinates at 180 degrees
        longitude_colname: 'str'
            specifiy column name that has longitude axis, default of "GAL-LON"
        colorbar: 'bool'
            if True, adds colorbar
        clip_flux: 'u.Quantity'
            flux value to clip max values of flux at
        label_axes: 'bool'
            if True, will label x and y axis
        sigma: 'float'
            gaussian filter smoothing factor if kind is "contour"
            default of 2
        auto_level_factors: 'list-like'
            [low,high] percentiles to use when auto computing levels for contour
            should be between 0 and 1
        n_levels: 'int'
            number of levels to use if autocalculating levels for contour
        """
        if not hasattr(ax, 'scatter'):
            if not hasattr(fig, 'add_subplot'):
                fig = plt.figure(constrained_layout = True)
                ax = fig.add_subplot(111)
            else:
                ax = fig.add_subplot(111)
        else:
            if not hasattr(fig, 'add_subplot'):
                fig = ax.get_figure()

        # Filter by galactic coordinate ranges
        if bounds is None:
            if (lrange is None) & (brange is None):
                tab_filtered = self
            else:
                if lrange is None:
                    lrange = [-180,180]*u.deg
                elif brange is None:
                    brange = [-90,90]*u.deg

                if not hasattr(lrange, "unit"):
                    lrange*=u.deg
                if not hasattr(brange, "unit"):
                    brange*=u.deg

                bounds = [*lrange, *brange]

                tab_filtered = self.sky_section(bounds = bounds, radius = radius, wrap_at_180 = wrap_at_180)
        else:
            tab_filtered = self.sky_section(bounds = bounds, radius = radius, wrap_at_180 = wrap_at_180)

        flux, velocity = tab_filtered.get_snr_masked_intensity(colname, 
                                                       snr_cut = snr_cut, 
                                                       rayleigh = rayleigh, 
                                                       velocity = True)
        if vel_diff_colname is not None:
            flux_vd, velocity_vd = tab_filtered.get_snr_masked_intensity(vel_diff_colname, 
                                                       snr_cut = snr_cut, 
                                                       rayleigh = rayleigh, 
                                                       velocity = True)

        if velocity_bounds is None:
            velocity_bounds = [-200, 200] * u.km/u.s 
        if not hasattr(velocity_bounds, "unit"):
            velocity_bounds *= u.km/u.s 
            logging.warning("No units specified for velocity_bounds - assuming km/s")

        vel_mask = velocity < velocity_bounds[0]
        vel_mask |= velocity > velocity_bounds[1]
        flux.mask = np.logical_or(flux.mask,vel_mask) 
        velocity.mask = np.logical_or(velocity.mask,vel_mask) 

        if vel_diff_colname is not None:
            vel_mask_vd = velocity_vd < velocity_bounds[0]
            vel_mask_vd |= velocity_vd > velocity_bounds[1]
            flux_vd.mask = np.logical_or(flux_vd.mask,vel_mask_vd) 
            velocity_vd.mask = np.logical_or(velocity_vd.mask,vel_mask_vd) 

        if clip_flux is not None:
            if not hasattr(clip_flux, "unit"):
                clip_flux *= flux.unit
            flux.mask = np.logical_or(flux.mask, flux > clip_flux)

        lon = Angle(tab_filtered[longitude_colname])
        if wrap_at_180:
            wrap_at = "180d"
        else:
            wrap_at = "360d"
        lon = lon.wrap_at(wrap_at)
        lon = Masked(lon, mask = flux.mask)

        valid_mask = ~flux.mask

    

        if vel_diff_colname is not None:
            valid_mask_vd = ~flux_vd.mask 
            valid_mask_vd &= valid_mask
            # compute weighted histogram data
            # Flux Weight
            H_flux, xedges, yedges = np.histogram2d(
                lon.unmasked[valid_mask_vd].value,
                velocity.unmasked[valid_mask_vd].value,
                weights = flux.unmasked.value[valid_mask_vd],
                bins=gridsize,
                density=False,
                )

            #Flux * Velocity difference weight
            H_vd, xedges, yedges = np.histogram2d(
                lon.unmasked[valid_mask_vd].value,
                velocity.unmasked[valid_mask_vd].value,
                weights = flux.unmasked.value[valid_mask_vd] * (
                        velocity.unmasked[valid_mask_vd].value - 
                        velocity_vd.unmasked[valid_mask_vd].value
                    ),
                bins=gridsize,
                density=False,
                )

            H_vd_flux_weighted = H_vd / H_flux

            hb = ax.pcolormesh(xedges, yedges, H_vd_flux_weighted.T, 
                                norm = kwargs.pop("norm", Normalize(vmin = -10, vmax = 10)),
                                cmap = kwargs.pop("cmap", "RdBu_r"),
                                **kwargs)



        else:

            if kind == "hex":
                hb = ax.hexbin(lon[valid_mask], velocity[valid_mask], 
                               gridsize = gridsize,
                               bins = bins, 
                               C = flux.value[valid_mask],
                               norm = kwargs.pop("norm",LogNorm(vmin = 1e0, vmax = 1e3)),
                               **kwargs)

            if kind in ["contour", "contourf"]:
                # Make 2D density histogram
                H, xedges, yedges = np.histogram2d(
                    lon.unmasked[valid_mask].value,
                    velocity.unmasked[valid_mask].value,
                    weights = flux.unmasked.value[valid_mask],
                    bins=gridsize,
                    density=False,
                )

                # Smooth the density field
                H_smooth = gaussian_filter(H, sigma=sigma)

                # Convert bin edges to coordinates
                X, Y = np.meshgrid(
                    (xedges[:-1] + xedges[1:]) / 2,
                    (yedges[:-1] + yedges[1:]) / 2
                )

                if auto_level_factors is None:
                    auto_level_factors = [.1,.80]
                levels = kwargs.pop("levels", np.geomspace(
                H_smooth.max()*auto_level_factors[0],
                H_smooth.max()*auto_level_factors[1],
                n_levels),
                )

                hb = ax.contour(X, Y, H_smooth.T, 
                                alpha = kwargs.pop("alpha", 0.7),
                                # norm = kwargs.pop("norm", LogNorm(vmin = 1e0, vmax = 1e3)),
                                levels = levels,
                                **kwargs)
        if colorbar:
            if vel_diff_colname is not None:
                cb = fig.colorbar(hb, label = cbar_kwargs.pop("label",r"$\Delta~V$ [{}$-${}] ({})".format(colname[5:], 
                                                        vel_diff_colname[5:], 
                                                        velocity.unit)),
                    **cbar_kwargs)

            else:

                try:
                    cb = fig.colorbar(hb, label = cbar_kwargs.pop("label","{} ({})".format(colname, flux.unit)),
                    **cbar_kwargs)

                except ValueError:
                    cb = fig.colorbar(hb, **cbar_kwargs)

        if label_axes:
            _ = ax.set_xlabel("Galactic Longitude (deg)", fontsize = 12)
            _ = ax.set_ylabel(r"$V_{{LSR}}$ ($km~s^{{-1}}$)", fontsize = 12)

        return fig




    def plot_bv(self,
                colname,
                vel_diff_colname = None,
                fig = None,
                ax = None,
                lrange = None,
                brange = None,
                bounds = None,
                radius = None,
                snr_cut = 5,
                kind='hex',
                gridsize=50,
                bins = "log",
                rayleigh = True,
                velocity_bounds = None,
                latitude_colname = "GAL-LAT",
                colorbar = True,
                clip_flux = None,
                label_axes = True,
                wrap_at_180 = True,
                sigma = 2,
                auto_level_factors = None,
                n_levels = 6,
                cbar_kwargs = {},
                **kwargs
                ):
        """
        Plot latitude vs. velocity diagram for specified flux column

        Parameters
        ----------
        colname: 'str'
            column name for flux to use for plot
        vel_diff_colname: 'str', optional, must be keyword
            if provided will colormap the lv diagram by intensity weighted velocity difference
            intensity weighting is done with colname
            velocity difference is from colname - vel_diff_colname
        fig: 'plt.figure', optional, must be keyword
            if provided, will create axes on the figure provided
        ax: 'plt.figure.axes' or 'cartopy.axes`, optional, must be keyword
            if provided, will plot on these axes
            can provide ax as a cartpy projection that contains different map projections
        lrange (tuple): Optional 
            (min, max) Galactic longitude range.
        brange (tuple): Optional 
            (min, max) Galactic latitude range.
        bounds: `list` or `Quantity` or `SkyCoord`
            if provided, will ignore l_range and b_range
            if `list` or `Quantity` must be formatted as:
                [min Galactic Longitude, max Galactic Longitude, min Galactic Latitude, max Galactic Latitude]
                or 
                [center Galactic Longitude, center Galactic Latitude] and requires radius keyword to be set
                default units of u.deg are assumed
            if `SkyCoord', must be length 4 or length 1 or length 2
                length 4 specifies 4 corners of rectangular shape
                length 1 specifies center of circular region and requires radius keyword to be set
                length 2 specifies two corners of rectangular region
        radius: 'number' or 'Quantity', optional, must be keyword
            sets radius of circular region
            only used if bounds is provided and is 2d
        snr_cut: 'float'
            Signal to Noise cut to impose on data - default of 3
        kind: 'str',
            kind of plot to make ["scatter", "hex", etc]
        gridsize: 'int'
            number of bins to use for histogramming
        bins: 'str'
            scaling to use for colormapping - default of log
        rayleigh: 'bool'
            if True, converts flux to rayleighs
        velocity_bounds: 'list-like'
            [min,max] velocity to use for velocity axis
        latitude_colname: 'str'
            specifiy column name that has longitude axis, default of "GAL-LON"
        colorbar: 'bool'
            if True, adds colorbar
        clip_flux: 'u.Quantity'
            flux value to clip max values of flux at
        label_axes: 'bool'
            if True, will label x and y axis
        wrap_at_180: 'bool'
            if True, wraps coordinates at 180 degrees
        sigma: 'float'
            gaussian filter smoothing factor if kind is "contour"
            default of 2
        auto_level_factors: 'list-like'
            [low,high] percentiles to use when auto computing levels for contour
            should be between 0 and 1
        n_levels: 'int'
            number of levels to use if autocalculating levels for contour
        """
        if not hasattr(ax, 'scatter'):
            if not hasattr(fig, 'add_subplot'):
                fig = plt.figure(constrained_layout = True)
                ax = fig.add_subplot(111)
            else:
                ax = fig.add_subplot(111)
        else:
            if not hasattr(fig, 'add_subplot'):
                fig = ax.get_figure()

        # Filter by galactic coordinate ranges
        if bounds is None:
            if (lrange is None) & (brange is None):
                tab_filtered = self
            else:
                if lrange is None:
                    lrange = [-180,180]*u.deg
                elif brange is None:
                    brange = [-90,90]*u.deg

                if not hasattr(lrange, "unit"):
                    lrange*=u.deg
                if not hasattr(brange, "unit"):
                    brange*=u.deg

                bounds = [*lrange, *brange]

                tab_filtered = self.sky_section(bounds = bounds, radius = radius, wrap_at_180 = wrap_at_180)
        else:
            tab_filtered = self.sky_section(bounds = bounds, radius = radius, wrap_at_180 = wrap_at_180)

        flux, velocity = tab_filtered.get_snr_masked_intensity(colname, 
                                                       snr_cut = snr_cut, 
                                                       rayleigh = rayleigh, 
                                                       velocity = True)
        if vel_diff_colname is not None:
            flux_vd, velocity_vd = tab_filtered.get_snr_masked_intensity(vel_diff_colname, 
                                                       snr_cut = snr_cut, 
                                                       rayleigh = rayleigh, 
                                                       velocity = True)

        if velocity_bounds is None:
            velocity_bounds = [-200, 200] * u.km/u.s 
        if not hasattr(velocity_bounds, "unit"):
            velocity_bounds *= u.km/u.s 
            logging.warning("No units specified for velocity_bounds - assuming km/s")

        vel_mask = velocity < velocity_bounds[0]
        vel_mask |= velocity > velocity_bounds[1]
        flux.mask = np.logical_or(flux.mask,vel_mask) 
        velocity.mask = np.logical_or(velocity.mask,vel_mask) 

        if vel_diff_colname is not None:
            vel_mask_vd = velocity_vd < velocity_bounds[0]
            vel_mask_vd |= velocity_vd > velocity_bounds[1]
            flux_vd.mask = np.logical_or(flux_vd.mask,vel_mask_vd) 
            velocity_vd.mask = np.logical_or(velocity_vd.mask,vel_mask_vd) 

        if clip_flux is not None:
            if not hasattr(clip_flux, "unit"):
                clip_flux *= flux.unit
            flux.mask = np.logical_or(flux.mask, flux > clip_flux)

        lat = Angle(tab_filtered[latitude_colname])
        lat = Masked(lat, mask = flux.mask)

        valid_mask = ~flux.mask

    

        if vel_diff_colname is not None:
            valid_mask_vd = ~flux_vd.mask 
            valid_mask_vd &= valid_mask
            # compute weighted histogram data
            # Flux Weight
            H_flux, xedges, yedges = np.histogram2d(
                lat.unmasked[valid_mask_vd].value,
                velocity.unmasked[valid_mask_vd].value,
                weights = flux.unmasked.value[valid_mask_vd],
                bins=gridsize,
                density=False,
                )

            #Flux * Velocity difference weight
            H_vd, xedges, yedges = np.histogram2d(
                lat.unmasked[valid_mask_vd].value,
                velocity.unmasked[valid_mask_vd].value,
                weights = flux.unmasked.value[valid_mask_vd] * (
                        velocity.unmasked[valid_mask_vd].value - 
                        velocity_vd.unmasked[valid_mask_vd].value
                    ),
                bins=gridsize,
                density=False,
                )

            H_vd_flux_weighted = H_vd / H_flux

            hb = ax.pcolormesh(xedges, yedges, H_vd_flux_weighted.T, 
                                norm = kwargs.pop("norm", Normalize(vmin = -10, vmax = 10)),
                                cmap = kwargs.pop("cmap", "RdBu_r"),
                                **kwargs)



        else:

            if kind == "hex":
                hb = ax.hexbin(lat[valid_mask], velocity[valid_mask], 
                               gridsize = gridsize,
                               bins = bins, 
                               C = flux.value[valid_mask],
                               norm = kwargs.pop("norm",LogNorm(vmin = 1e0, vmax = 1e3)),
                               **kwargs)

            if kind in ["contour", "contourf"]:
                # Make 2D density histogram
                H, xedges, yedges = np.histogram2d(
                    lat.unmasked[valid_mask].value,
                    velocity.unmasked[valid_mask].value,
                    weights = flux.unmasked.value[valid_mask],
                    bins=gridsize,
                    density=False,
                )

                # Smooth the density field
                H_smooth = gaussian_filter(H, sigma=sigma)

                # Convert bin edges to coordinates
                X, Y = np.meshgrid(
                    (xedges[:-1] + xedges[1:]) / 2,
                    (yedges[:-1] + yedges[1:]) / 2
                )

                if auto_level_factors is None:
                    auto_level_factors = [.1,.80]
                levels = kwargs.pop("levels", np.geomspace(
                H_smooth.max()*auto_level_factors[0],
                H_smooth.max()*auto_level_factors[1],
                n_levels),
                )

                hb = ax.contour(X, Y, H_smooth.T, 
                                alpha = kwargs.pop("alpha", 0.7),
                                # norm = kwargs.pop("norm", LogNorm(vmin = 1e0, vmax = 1e3)),
                                levels = levels,
                                **kwargs)
        if colorbar:
            if vel_diff_colname is not None:
                cb = fig.colorbar(hb, label = cbar_kwargs.pop("label",r"$\Delta~V$ [{}$-${}] ({})".format(colname[5:], 
                                                        vel_diff_colname[5:], 
                                                        velocity.unit)),
                    **cbar_kwargs)

            else:

                try:
                    cb = fig.colorbar(hb, label = cbar_kwargs.pop("label","{} ({})".format(colname, flux.unit)),
                    **cbar_kwargs)

                except ValueError:
                    cb = fig.colorbar(hb, **cbar_kwargs)

        if label_axes:
            _ = ax.set_xlabel("Galactic Latitude (deg)", fontsize = 12)
            _ = ax.set_ylabel(r"$V_{{LSR}}$ ($km~s^{{-1}}$)", fontsize = 12)

        return fig






