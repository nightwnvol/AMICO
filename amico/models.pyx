# cython: language_level = 3
# distutils: language = c++

from __future__ import absolute_import, division, print_function

import numpy as np
import scipy
from os.path import join as pjoin
import amico.lut
from tqdm import tqdm
import abc
from amico.util import PRINT, ERROR, get_verbose
from amico.synthesis import Stick, Zeppelin, Ball, CylinderGPD, SphereGPD, Astrosticks, NODDIIntraCellular, NODDIExtraCellular, NODDIIsotropic
from joblib import cpu_count, Parallel, delayed
import time

cimport cython
from libc.stdlib cimport malloc, free
from libc.math cimport pi, atan2

import warnings
warnings.filterwarnings("ignore") # needed for a problem with spams
warnings.formatwarning = lambda message, category, filename, lineno, line=None: \
    "[WARNING] %s " % message

# import the spams module, which is used only to fit the models in AMICO.
# But, on the other hand, using the models from COMMIT does not require that!
try :
    import spams
except ImportError:
    warnings.warn('Module "spams" does not seems to be installed; perhaps you will not be able to call the fit() functions of some models.')


class BaseModel( object ) :
#class BaseModel( object, metaclass=abc.ABCMeta ) :
    """Basic class to build a model; new models should inherit from this class.
    All the methods need to be overloaded to account for the specific needs of the model.
    Each method will then be called by a dispatcher when needed.

    NB: this model also serves the purpose of illustrating the creation of new models.

    Attributes
    ----------
    id : string
        Identification code for the model
    name : string
        A more human-readable description for the model (can be equal to id)
    scheme: Scheme class
        Acquisition scheme to be used for resampling
    maps_name : list of strings
        Names of the maps computed/returned by the model (suffix to saved filenames)
    maps_descr : list of strings
        Description of each map (will be saved in the description of the NIFTI header)
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__( self ) :
        """To define the parameters of the model, e.g. id and name, returned maps,
        model-specific parameters etc.
        """
        self.id         = 'BaseModel'
        self.name       = 'Base Model'
        self.maps_name  = []
        self.maps_descr = []
        self.scheme = None
        return


    @abc.abstractmethod
    def set( self, *args, **kwargs ) :
        """For setting all the parameters specific to the model.
        NB: the parameters are model-dependent.
        """
        return


    @abc.abstractmethod
    def get_params( self ) :
        """For getting the actual values of all the parameters specific to the model.
        NB: the parameters are model-dependent.
        """
        return


    @abc.abstractmethod
    def set_solver( self, *args, **kwargs ) :
        """For setting the parameters required by the solver to fit the model.
        NB: the parameters are model-dependent.

        Returns
        -------
        params : dictionary
            All the parameters that the solver will need to fit the model
        """
        return


    @abc.abstractmethod
    def generate( self, out_path, aux, idx_in, idx_out, ndirs ) :
        """For generating the signal response-functions and createing the LUT.
        NB: do not change the signature!

        Parameters
        ----------
        out_path : string
            Path where the response function have to be saved
        aux : structure
            Auxiliary structures to perform SH fitting and rotations
        idx_in : array
            Indices of the samples belonging to each shell
        idx_out : array
            Indices of the SH coefficients corresponding to each shell
        ndirs : int
            Number of directions on the half of the sphere representing the possible orientations of the response functions
        """
        return


    @abc.abstractmethod
    def resample( self, in_path, idx_out, Ylm_out, doMergeB0, ndirs ) :
        """For projecting the LUT to the subject space.
        NB: do not change the signature!

        Parameters
        ----------
        in_path : Scheme class
            Acquisition scheme of the acquired signal
        idx_out : array
            Indices of the samples belonging to each shell
        Ylm_out : array
            SH bases to project back each shell to signal space
        doMergeB0: bool
            Merge b0-volumes into a single volume if True
        ndirs : int
            Number of directions on the half of the sphere representing the possible orientations of the response functions

        Returns
        -------
        KERNELS : dictionary
            Contains the LUT and all corresponding details. In particular, it is
            required to have a field 'model' set to "self.id".
        """
        return


    @abc.abstractmethod
    def fit( self, y, dirs, KERNELS, params ) :
        """For fitting the model to the data.
        NB: do not change the signature!

        Parameters
        ----------
        y : array
            Diffusion signal at this voxel
        dirs : list of arrays
            Directions fitted in the voxel
        KERNELS : dictionary
            Contains the LUT and all corresponding details
        params : dictionary
            Parameters to be used by the solver

        Returns
        -------
        MAPs : list of floats
            Scalar values eastimated in each voxel
        dirs_mod : list of arrays
            Updated directions (if applicable), otherwise just return dirs
        x : array
            Coefficients of the fitting
        A : array
            Actual dictionary used in the fitting
        """
        return



class StickZeppelinBall( BaseModel ) :
    """Implements the Stick-Zeppelin-Ball model [1].

    The intra-cellular contributions from within the axons are modeled as "sticks", i.e.
    tensors with a given axial diffusivity (d_par) but null perpendicular diffusivity (d_perp=0);
    if d_perp>0, then a Zeppelin is used instead of a Stick.
    Extra-cellular contributions are modeled as "Zeppelins", i.e. tensors with a given axial
    diffusivity (d_par_zep) and, possibily, a series of perpendicular diffusivities (d_perps_zep).
    If the axial diffusivity of the Zeppelins is not specified, then it is assumed equal to that
    of the Stick. Isotropic contributions are modeled as "Balls", i.e. tensors with isotropic
    diffusivities (d_isos).

    References
    ----------
    .. [1] Panagiotaki et al. (2012) Compartment models of the diffusion MR signal
           in brain white matter: A taxonomy and comparison. NeuroImage, 59: 2241-54
    """

    def __init__( self ) :
        self.id         = 'StickZeppelinBall'
        self.name       = 'Stick-Zeppelin-Ball'
        self.maps_name  = [ ]
        self.maps_descr = [ ]

        self.d_par       = 1.7E-3                                          # Parallel diffusivity for the Stick [mm^2/s]
        self.d_perp      = 0                                               # Perpendicular diffusivity for the Stick [mm^2/s]
        self.d_par_zep   = 1.7E-3                                          # Parallel diffusivity for the Zeppelins [mm^2/s]
        self.d_perps_zep = np.array([ 1.19E-3, 0.85E-3, 0.51E-3, 0.17E-3]) # Perpendicular diffusivitie(s) [mm^2/s]
        self.d_isos      = np.array([ 3.0E-3 ])                            # Isotropic diffusivitie(s) [mm^2/s]


    def set( self, d_par, d_perps_zep, d_isos, d_par_zep=None, d_perp=0 ) :
        self.d_par = d_par
        self.d_perp = d_perp
        if d_par_zep is None:
            self.d_par_zep = d_par
        else:
            self.d_par_zep = d_par_zep
        self.d_perps_zep = np.array( d_perps_zep )
        self.d_isos  = np.array( d_isos )


    def get_params( self ) :
        params = {}
        params['id'] = self.id
        params['name'] = self.name
        params['d_par'] = self.d_par
        params['d_perp'] = self.d_perp
        params['d_par_zep'] = self.d_par_zep
        params['d_perps_zep'] = self.d_perps_zep
        params['d_isos'] = self.d_isos
        return params


    def set_solver( self ) :
        ERROR( 'Not implemented' )


    def generate( self, out_path, aux, idx_in, idx_out, ndirs ) :
        scheme_high = amico.lut.create_high_resolution_scheme( self.scheme )

        stick = Stick(scheme_high)
        zeppelin = Zeppelin(scheme_high)
        ball = Ball(scheme_high)

        nATOMS = 1 + len(self.d_perps_zep) + len(self.d_isos)
        idx = 0
        with tqdm(total=nATOMS, ncols=70, bar_format='   |{bar}| {percentage:4.1f}%', disable=(get_verbose()<3)) as progress:
            # Stick
            signal = stick.get_signal(self.d_par)
            lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, False, ndirs )
            np.save( pjoin( out_path, f'A_{idx+1:03d}.npy' ), lm )
            idx += 1
            progress.update()
            # Zeppelin(s)
            for d in self.d_perps_zep :
                signal = zeppelin.get_signal(self.d_par_zep, d)
                lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, False, ndirs )
                np.save( pjoin( out_path, f'A_{idx+1:03d}.npy' ), lm )
                idx += 1
                progress.update()
            # Ball(s)
            for d in self.d_isos :
                signal = ball.get_signal(d)
                lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, True, ndirs )
                np.save( pjoin( out_path, f'A_{idx+1:03d}.npy' ), lm )
                idx += 1
                progress.update()


    def resample( self, in_path, idx_out, Ylm_out, doMergeB0, ndirs ) :
        KERNELS = {}
        KERNELS['model'] = self.id
        if doMergeB0:
            nS = 1+self.scheme.dwi_count
            merge_idx = np.hstack((self.scheme.b0_idx[0],self.scheme.dwi_idx))
        else:
            nS = self.scheme.nS
            merge_idx = np.arange(nS)
        KERNELS['wmr']   = np.zeros( (1,ndirs,nS), dtype=np.float32 )
        KERNELS['wmh']   = np.zeros( (len(self.d_perps_zep),ndirs,nS), dtype=np.float32 )
        KERNELS['iso']   = np.zeros( (len(self.d_isos),nS), dtype=np.float32 )

        nATOMS = 1 + len(self.d_perps_zep) + len(self.d_isos)
        idx = 0
        with tqdm(total=nATOMS, ncols=70, bar_format='   |{bar}| {percentage:4.1f}%', disable=(get_verbose()<3)) as progress:
            # Stick
            lm = np.load( pjoin( in_path, f'A_{idx+1:03d}.npy' ) )
            if lm.shape[0] != ndirs:
                ERROR( 'Outdated LUT. Call "generate_kernels( regenerate=True )" to update the LUT' )
            KERNELS['wmr'][0,...] = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, False, ndirs )[:,merge_idx]
            idx += 1
            progress.update()

            # Zeppelin(s)
            for i in range(len(self.d_perps_zep)) :
                lm = np.load( pjoin( in_path, f'A_{idx+1:03d}.npy' ) )
                if lm.shape[0] != ndirs:
                    ERROR( 'Outdated LUT. Call "generate_kernels( regenerate=True )" to update the LUT' )
                KERNELS['wmh'][i,...] = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, False, ndirs )[:,merge_idx]
                idx += 1
                progress.update()

            # Ball(s)
            for i in range(len(self.d_isos)) :
                lm = np.load( pjoin( in_path, f'A_{idx+1:03d}.npy' ) )
                KERNELS['iso'][i,...] = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, True, ndirs )[merge_idx]
                idx += 1
                progress.update()

        return KERNELS


    def fit( self, y, dirs, KERNELS, params ) :
        ERROR( 'Not implemented' )



class CylinderZeppelinBall( BaseModel ) :
    """Implements the Cylinder-Zeppelin-Ball model [1].

    The intra-cellular contributions from within the axons are modeled as "cylinders"
    with specific radii (Rs) and a given axial diffusivity (d_par).
    Extra-cellular contributions are modeled as tensors with the same axial diffusivity
    as the cylinders (d_par) and, possibily, a series of perpendicular diffusivities (d_perps).
    Isotropic contributions are modeled as tensors with isotropic diffusivities (d_isos).

    NB: this model works only with schemes containing the full specification of
        the diffusion gradients (eg gradient strength, small delta etc).

    References
    ----------
    .. [1] Panagiotaki et al. (2012) Compartment models of the diffusion MR signal
           in brain white matter: A taxonomy and comparison. NeuroImage, 59: 2241-54
    """

    def __init__( self ) :
        self.id         = 'CylinderZeppelinBall'
        self.name       = 'Cylinder-Zeppelin-Ball'
        self.maps_name  = [ 'v', 'a', 'd' ]
        self.maps_descr = [ 'Intra-cellular volume fraction', 'Mean axonal diameter', 'Axonal density' ]

        self.d_par   = 0.6E-3                                                    # Parallel diffusivity [mm^2/s]
        self.Rs      = np.concatenate( ([0.01],np.linspace(0.5,8.0,20)) ) * 1E-6 # Radii of the axons [meters]
        self.d_perps = np.array([ 1.19E-3, 0.85E-3, 0.51E-3, 0.17E-3])           # Perpendicular diffusivitie(s) [mm^2/s]
        self.d_isos  = np.array( [ 2.0E-3 ] )                                    # Isotropic diffusivitie(s) [mm^2/s]
        self.isExvivo  = False                                                   # Add dot compartment to dictionary (exvivo data)


    def set( self, d_par, Rs, d_perps, d_isos ) :
        self.d_par   = d_par
        self.Rs      = np.array(Rs)
        self.d_perps = np.array(d_perps)
        self.d_isos  = np.array(d_isos)


    def get_params( self ) :
        params = {}
        params['id'] = self.id
        params['name'] = self.name
        params['d_par'] = self.d_par
        params['Rs'] = self.Rs
        params['d_perps'] = self.d_perps
        params['d_isos'] = self.d_isos
        params['isExvivo'] = self.isExvivo
        return params


    def set_solver( self, lambda1 = 0.0, lambda2 = 4.0 ) :
        params = {}
        params['mode']    = 2
        params['pos']     = True
        params['lambda1'] = lambda1
        params['lambda2'] = lambda2
        return params


    def generate( self, out_path, aux, idx_in, idx_out, ndirs ) :
        if self.scheme.version != 1 :
            ERROR( 'This model requires a "VERSION: STEJSKALTANNER" scheme' )

        scheme_high = amico.lut.create_high_resolution_scheme( self.scheme )

        cylinder = CylinderGPD(scheme_high)
        zeppelin = Zeppelin(scheme_high)
        ball = Ball(scheme_high)

        nATOMS = len(self.Rs) + len(self.d_perps) + len(self.d_isos)
        idx = 0
        with tqdm(total=nATOMS, ncols=70, bar_format='   |{bar}| {percentage:4.1f}%', disable=(get_verbose()<3)) as progress:
            # Cylinder(s)
            for R in self.Rs :
                signal = cylinder.get_signal(self.d_par, R)
                lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, False, ndirs )
                np.save( pjoin( out_path, f'A_{idx+1:03d}.npy' ), lm )
                idx += 1
                progress.update()

            # Zeppelin(s)
            for d in self.d_perps :
                signal = zeppelin.get_signal(self.d_par, d)
                lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, False, ndirs )
                np.save( pjoin( out_path, f'A_{idx+1:03d}.npy' ), lm )
                idx += 1
                progress.update()

            # Ball(s)
            for d in self.d_isos :
                signal = ball.get_signal(d)
                lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, True, ndirs )
                np.save( pjoin( out_path, f'A_{idx+1:03d}.npy' ), lm )
                idx += 1
                progress.update()


    def resample( self, in_path, idx_out, Ylm_out, doMergeB0, ndirs ) :
        if doMergeB0:
            nS = 1+self.scheme.dwi_count
            merge_idx = np.hstack((self.scheme.b0_idx[0],self.scheme.dwi_idx))
        else:
            nS = self.scheme.nS
            merge_idx = np.arange(nS)
        KERNELS = {}
        KERNELS['model'] = self.id
        KERNELS['wmr'] = np.zeros( (len(self.Rs),ndirs,nS,), dtype=np.float32 )
        KERNELS['wmh'] = np.zeros( (len(self.d_perps),ndirs,nS,), dtype=np.float32 )
        KERNELS['iso'] = np.zeros( (len(self.d_isos),nS,), dtype=np.float32 )

        nATOMS = len(self.Rs) + len(self.d_perps) + len(self.d_isos)
        idx = 0
        with tqdm(total=nATOMS, ncols=70, bar_format='   |{bar}| {percentage:4.1f}%', disable=(get_verbose()<3)) as progress:
            # Cylinder(s)
            for i in range(len(self.Rs)) :
                lm = np.load( pjoin( in_path, f'A_{idx+1:03d}.npy' ) )
                if lm.shape[0] != ndirs:
                    ERROR( 'Outdated LUT. Call "generate_kernels( regenerate=True )" to update the LUT' )
                KERNELS['wmr'][i,:,:] = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, False, ndirs )[:,merge_idx]
                idx += 1
                progress.update()

            # Zeppelin(s)
            for i in range(len(self.d_perps)) :
                lm = np.load( pjoin( in_path, f'A_{idx+1:03d}.npy' ) )
                if lm.shape[0] != ndirs:
                    ERROR( 'Outdated LUT. Call "generate_kernels( regenerate=True )" to update the LUT' )
                KERNELS['wmh'][i,:,:] = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, False, ndirs )[:,merge_idx]
                idx += 1
                progress.update()

            # Ball(s)
            for i in range(len(self.d_isos)) :
                lm = np.load( pjoin( in_path, f'A_{idx+1:03d}.npy' ) )
                KERNELS['iso'][i,:] = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, True, ndirs )[merge_idx]
                idx += 1
                progress.update()

        return KERNELS


    def fit( self, y, dirs, KERNELS, params, htable ) :
        nD = dirs.shape[0]
        n1 = len(self.Rs)
        n2 = len(self.d_perps)
        n3 = len(self.d_isos)
        if self.isExvivo:
            nATOMS = nD*(n1+n2)+n3+1
        else:
            nATOMS = nD*(n1+n2)+n3
        # prepare DICTIONARY from dirs and lookup tables
        A = np.ones( (len(y), nATOMS ), dtype=np.float64, order='F' )
        o = 0
        for i in range(nD) :
            lut_idx = amico.lut.dir_TO_lut_idx( dirs[i], htable )
            A[:,o:(o+n1)] = KERNELS['wmr'][:,lut_idx,:].T
            o += n1
        for i in range(nD) :
            lut_idx = amico.lut.dir_TO_lut_idx( dirs[i], htable )
            A[:,o:(o+n2)] = KERNELS['wmh'][:,lut_idx,:].T
            o += n2
        A[:,o:] = KERNELS['iso'].T

        # empty dictionary
        if A.shape[1] == 0 :
            return [0, 0, 0], None, None, None

        # fit
        x = spams.lasso( np.asfortranarray( y.reshape(-1,1) ), D=A, numThreads=1, **params ).todense().A1

        # return estimates
        f1 = x[ :(nD*n1) ].sum()
        f2 = x[ (nD*n1):(nD*(n1+n2)) ].sum()
        v = f1 / ( f1 + f2 + 1e-16 )
        xIC = x[:nD*n1].reshape(-1,n1).sum(axis=0)
        a = 1E6 * 2.0 * np.dot(self.Rs,xIC) / ( f1 + 1e-16 )
        d = (4.0*v) / ( np.pi*a**2 + 1e-16 )
        return [v, a, d], dirs, x, A


cdef extern from 'wrappers.h':
    cdef void nnls(const double *A, const double *y, const int m, const int n, double *x, double *rnorm) nogil
    cdef void lasso(double *A, double *y, const int m, const int n, const double lambda1, const double lambda2, double *x) nogil

cdef class NODDI:
    """Implements the NODDI model [2].

    NB: this model does not require to have the "NODDI MATLAB toolbox" installed;
        all the necessary functions have been ported to Python.

    References
    ----------
    .. [2] Zhang et al. (2012) NODDI: Practical in vivo neurite orientation
           dispersion and density imaging of the human brain. NeuroImage, 61: 1000-16
    """
    cdef object id
    cdef object name
    cdef object maps_name
    cdef object maps_descr
    cdef object dPar
    cdef object dIso
    cdef object IC_VFs
    cdef object IC_ODs
    cdef object isExvivo

    cdef object scheme

    def __cinit__( self ):
        self.id         = "NODDI"
        self.name       = "NODDI"
        self.maps_name  = [ 'ICVF', 'OD', 'ISOVF' ]
        self.maps_descr = [ 'Intra-cellular volume fraction', 'Orientation dispersion', 'Isotropic volume fraction' ]

        self.dPar      = 1.7E-3
        self.dIso      = 3.0E-3
        self.IC_VFs    = np.linspace(0.1, 0.99, 12)
        self.IC_ODs    = np.hstack((np.array([0.03, 0.06]), np.linspace(0.09, 0.99, 10)))
        self.isExvivo  = False

    @property
    def id(self):
        return self.id
    @property
    def name(self):
        return self.name
    @property
    def maps_name(self):
        return self.maps_name
    @property
    def maps_descr(self):
        return self.maps_descr
    @property
    def dPar(self):
        return self.dPar
    @property
    def dIso(self):
        return self.dIso
    @property
    def IC_VFs(self):
        return self.IC_VFs
    @property
    def IC_ODs(self):
        return self.IC_ODs
    @property
    def isExvivo(self):
        return self.isExvivo

    @property
    def scheme(self):
        return self.scheme
    @scheme.setter
    def scheme(self, scheme):
        self.scheme = scheme

    def set( self, dPar, dIso, IC_VFs, IC_ODs, isExvivo ):
        self.dPar      = dPar
        self.dIso      = dIso
        self.IC_VFs    = np.array( IC_VFs )
        self.IC_ODs    = np.array( IC_ODs )
        self.isExvivo  = isExvivo
        if isExvivo:
            self.maps_name  = [ 'ICVF', 'OD', 'ISOVF', 'dot' ]
            self.maps_descr = [ 'Intra-cellular volume fraction', 'Orientation dispersion', 'Isotropic volume fraction', 'Dot volume fraction' ]
        else:
            self.maps_name  = [ 'ICVF', 'OD', 'ISOVF']
            self.maps_descr = [ 'Intra-cellular volume fraction', 'Orientation dispersion', 'Isotropic volume fraction']

    def get_params( self ) :
        params = {}
        params['id'] = self.id
        params['name'] = self.name
        params['dPar'] = self.dPar
        params['dIso'] = self.dIso
        params['IC_VFs'] = self.IC_VFs
        params['IC_ODs'] = self.IC_ODs
        params['isExvivo'] = self.isExvivo
        return params

    def set_solver( self, lambda1 = 5e-1, lambda2 = 1e-3 ):
        params = {}
        params['mode']    = 2
        params['pos']     = True
        params['lambda1'] = lambda1
        params['lambda2'] = lambda2
        return params

    def generate( self, out_path, aux, idx_in, idx_out, ndirs ):
        scheme_high = amico.lut.create_high_resolution_scheme( self.scheme )

        noddi_ic = NODDIIntraCellular(scheme_high)
        noddi_ec = NODDIExtraCellular(scheme_high)
        noddi_iso = NODDIIsotropic(scheme_high)

        nATOMS = len(self.IC_ODs)*len(self.IC_VFs) + 1
        idx = 0
        with tqdm(total=nATOMS, ncols=70, bar_format='   |{bar}| {percentage:4.1f}%', disable=(get_verbose()<3)) as progress:
            # Coupled contributions
            IC_KAPPAs = 1 / np.tan(self.IC_ODs*np.pi/2)
            for kappa in IC_KAPPAs:
                signal_ic = noddi_ic.get_signal(self.dPar, kappa)
                for v_ic in self.IC_VFs:
                    signal_ec = noddi_ec.get_signal(self.dPar, kappa, v_ic)
                    signal = v_ic*signal_ic + (1-v_ic)*signal_ec
                    lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, False, ndirs )
                    np.save( pjoin( out_path, f'A_{idx+1:03d}.npy') , lm )
                    idx += 1
                    progress.update()
            # Isotropic
            signal = noddi_iso.get_signal(self.dIso)
            lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, True, ndirs )
            np.save( pjoin( out_path, f'A_{nATOMS:03d}.npy') , lm )
            progress.update()

    def resample( self, in_path, idx_out, Ylm_out, doMergeB0, ndirs ):
        nATOMS = len(self.IC_ODs)*len(self.IC_VFs) + 1
        if doMergeB0:
            nS = 1+self.scheme.dwi_count
            merge_idx = np.hstack((self.scheme.b0_idx[0],self.scheme.dwi_idx))
        else:
            nS = self.scheme.nS
            merge_idx = np.arange(nS)
        KERNELS = {}
        KERNELS['model'] = self.id
        KERNELS['wm']    = np.zeros( (nATOMS-1,ndirs,nS), dtype=np.float32 )
        KERNELS['iso']   = np.zeros( nS, dtype=np.float32 )
        KERNELS['kappa'] = np.zeros( nATOMS-1, dtype=np.float32 )
        KERNELS['icvf']  = np.zeros( nATOMS-1, dtype=np.float32 )
        KERNELS['norms'] = np.zeros( (self.scheme.dwi_count, nATOMS-1) )

        idx = 0
        with tqdm(total=nATOMS, ncols=70, bar_format='   |{bar}| {percentage:4.1f}%', disable=(get_verbose()<3)) as progress:
            # Coupled contributions
            for i in range( len(self.IC_ODs) ):
                for j in range( len(self.IC_VFs) ):
                    lm = np.load( pjoin( in_path, f'A_{idx+1:03d}.npy' ) )
                    if lm.shape[0] != ndirs:
                        ERROR( 'Outdated LUT. Call "generate_kernels( regenerate=True )" to update the LUT' )
                    KERNELS['wm'][idx,:,:] = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, False, ndirs )[:,merge_idx]
                    KERNELS['kappa'][idx] = 1.0 / np.tan( self.IC_ODs[i]*np.pi/2.0 )
                    KERNELS['icvf'][idx]  = self.IC_VFs[j]
                    if doMergeB0:
                        KERNELS['norms'][:,idx] = 1 / np.linalg.norm( KERNELS['wm'][idx,0,1:] ) # norm of coupled atoms (for l1 minimization)
                    else:
                        KERNELS['norms'][:,idx] = 1 / np.linalg.norm( KERNELS['wm'][idx,0,self.scheme.dwi_idx] ) # norm of coupled atoms (for l1 minimization)
                    idx += 1
                    progress.update()
            # Isotropic
            lm = np.load( pjoin( in_path, f'A_{nATOMS:03d}.npy' ) )
            KERNELS['iso'] = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, True, ndirs )[merge_idx]
            progress.update()
        return KERNELS

    def fit(self, y, dirs, KERNELS, params, htable):
        # build chunks
        # NOTE make an helper function
        n_threads = cpu_count()
        n = y.shape[0]
        c = n // n_threads
        chunks = []
        for i, j in zip(range(0, n, c), range(c, n+1, c)):
            chunks.append((i, j))
        if chunks[-1][1] != n:
            chunks[-1] = (chunks[-1][0], n)

        # fit chunks in parallel
        results = Parallel(n_jobs=n_threads, prefer='threads')(delayed(self.fit_voxels)(y[i:j, :], dirs[i:j, :], KERNELS, params, htable) for i, j in chunks)
        # results = Parallel(n_jobs=n_threads, prefer='threads')(delayed(self.fit_voxels)(y[chunks[i][0]:chunks[i][1], :], dirs[chunks[i][0]:chunks[i][1], :], KERNELS, params, htable) for i in tqdm(range(len(chunks)), ncols=70, bar_format='   |{bar}| {percentage:4.1f}%', disable=(get_verbose()<3)))

        # return estimates
        estimates = []
        for result in results:
            estimates += result
        return estimates

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef fit_voxels(self, y, dirs, KERNELS, params, htable):
        cdef bint single_b0 = 1 if y.shape[1] == (1 + self.scheme.dwi_count) else 0

        cdef int nWM = len(self.IC_ODs) * len(self.IC_VFs)
        cdef int nATOMS = nWM + 1

        cdef bint is_exvivo = 1 if self.isExvivo else 0
        if is_exvivo:
            nATOMS += 1

        lut_idxs = np.array([amico.lut.dir_TO_lut_idx(d, htable) for d in dirs], dtype=np.intc)
        cdef int[::1] lut_idxs_view = lut_idxs

        cdef long long [::1]dwi_idx_view = self.scheme.dwi_idx

        cdef double [::1, :, :]K_wm_view = np.asfortranarray(np.swapaxes(KERNELS['wm'].T, 1, 2)).astype(np.double)
        cdef double [::1]K_iso_view = KERNELS['iso'].astype(np.double)
        cdef double [:, ::1]K_norms_view = KERNELS['norms'].astype(np.double) # NOTE C_CONTIGUOUS
        cdef double [::1]K_icvf_view = KERNELS['icvf'].astype(np.double)
        cdef double [::1]K_kappa_view = KERNELS['kappa'].astype(np.double)

        # nnls (1)
        # A
        K_wm_iso = np.asfortranarray(
            np.swapaxes(
                np.insert(
                    KERNELS['wm'],
                    KERNELS['wm'].shape[0],
                    KERNELS['iso'],
                    axis=0
                ).T,
                1,
                2
            )
        ).astype(np.double)
        cdef double [::1, :, :]K_wm_iso_view = K_wm_iso
        # y
        y = np.ascontiguousarray(y)
        cdef double [:, ::1]y_view = y
        # x
        x = np.zeros((nATOMS,), dtype=np.double) # NOTE nATOMS = K_wm_iso.shape[1]
        cdef double [::1]x_view = x
        # others
        cdef double r_norm = 0.0
        cdef double [::1]y_nnls_view = np.zeros((y.shape[1],), dtype=np.double)

        # lasso
        # A
        A2 = np.zeros(KERNELS['norms'].shape, dtype=np.double, order='F')
        cdef double [::1, :]A2_view = A2
        # y
        y2 = np.zeros((K_norms_view.shape[0],), dtype=np.double)
        cdef double [::1]y2_view = y2
        # x
        x2 = np.zeros((KERNELS['norms'].shape[1],), dtype=np.double)
        cdef double [::1]x2_view = x2
        # others
        cdef double lambda1 = params['lambda1']
        cdef double lambda2 = params['lambda2']
        
        # nnls (2)
        # A
        cdef double *A3
        # y
        # x
        cdef double *x3
        # others
        cdef int pos_count = 0
        cdef int *pos_idxs = <int *> malloc(sizeof(int) * nATOMS)

        # return
        cdef double v = 0.0
        cdef double od = 0.0
        cdef double fISO = 0.0
        est = np.zeros((y.shape[0], 3), dtype=np.double, order='F')
        cdef double [::1, :] est_view = est

        cdef double *xx = <double *> malloc(sizeof(double) * nATOMS)
        cdef double *x_wm = <double *> malloc(sizeof(double) * nWM)

        cdef double sum_nATOMS = 0.0
        cdef double sum_nWM = 0.0
        cdef double f1 = 0.0
        cdef double f2 = 0.0
        cdef double k1 = 0.0

        cdef Py_ssize_t i, j, k
        with nogil:
            for i in range(y_view.shape[0]):
                # CSF
                nnls(&K_wm_iso_view[0, 0, lut_idxs_view[i]], &y_view[i, 0], K_wm_iso_view.shape[0], K_wm_iso_view.shape[1], &x_view[0], &r_norm)
                
                for j in range(y_nnls_view.shape[0]):
                    y_nnls_view[j] = y_view[i, j] - x_view[nATOMS-1] * K_iso_view[j]
                    if is_exvivo:
                        y_nnls_view[j] = y_nnls_view[j] - x_view[nATOMS-2] * K_wm_view[j, nATOMS-1, lut_idxs_view[i]]
                    if y_nnls_view[j] < 0.0:
                        y_nnls_view[j] = 0.0


                # IC + EC
                if single_b0:
                    for j in range(A2_view.shape[0]):
                        for k in range(A2_view.shape[1]):
                            A2_view[j, k] = K_wm_view[j+1, k, lut_idxs_view[i]] * K_norms_view[j, k]
                        y2_view[j] = y_nnls_view[j+1]
                else:
                    for j in range(A2_view.shape[0]):
                        for k in range(A2_view.shape[1]):
                            A2_view[j, k] = K_wm_view[dwi_idx_view[j], k, lut_idxs_view[i]] * K_norms_view[j, k]
                        y2_view[j] = y_nnls_view[dwi_idx_view[j]]
                lasso(&A2_view[0, 0], &y2_view[0], A2_view.shape[0], A2_view.shape[1], lambda1, lambda2, &x2_view[0])


                # Debias coefficients
                pos_count = 0
                for j in range(x2_view.shape[0]):
                    x_view[j] = x2_view[j]
                    if x_view[j] > 0.0:
                        pos_idxs[pos_count] = j
                        pos_count += 1
                if is_exvivo:
                    x_view[nATOMS-2] = 1.0
                    pos_idxs[pos_count] = nATOMS-2
                    pos_count += 1    
                x_view[nATOMS-1] = 1.0
                pos_idxs[pos_count] = nATOMS-1
                pos_count += 1

                A3 = <double *> malloc(sizeof(double) * K_wm_iso_view.shape[0] * pos_count)
                for j in range(pos_count):
                    for k in range(K_wm_iso_view.shape[0]):
                        A3[j * K_wm_iso_view.shape[0] + k] = K_wm_iso_view[k, pos_idxs[j], lut_idxs_view[i]]

                x3 = <double *> malloc(sizeof(double) * pos_count)

                nnls(A3, &y_view[i, 0], K_wm_iso_view.shape[0], pos_count, x3, &r_norm)
                
                for j in range(pos_count):
                    x_view[pos_idxs[j]] = x3[j]
                free(x3)


                # Return estimates
                sum_nATOMS = 0.0
                sum_nWM = 0.0
                f1 = 0.0
                f2 = 0.0
                k1 = 0.0
                for j in range(nATOMS):
                    sum_nATOMS += x_view[j]
                for j in range(nWM):
                    sum_nWM += x_view[j]
                sum_nATOMS += 1e-16
                sum_nWM += 1e-16

                for j in range(nATOMS):
                    xx[j] = x_view[j] / sum_nATOMS

                for j in range(nWM):
                    x_wm[j] = x_view[j] / sum_nWM
                    f1 += K_icvf_view[j] * x_wm[j]
                    f2 += (1.0-K_icvf_view[j]) * x_wm[j]
                    k1 += K_kappa_view[j] * x_wm[j]
                    
                v = f1 / (f1 + f2 + 1e-16)
                od = 2.0 / pi * atan2(1.0, k1)
                fISO = fISO = xx[nATOMS-1]

                est_view[i, 0] = v
                est_view[i, 1] = od
                est_view[i, 2] = fISO

        free(pos_idxs)
        free(xx)
        free(x_wm)
        return est.tolist()
        


# class NODDI( BaseModel ) :
#     """Implements the NODDI model [2].

#     NB: this model does not require to have the "NODDI MATLAB toolbox" installed;
#         all the necessary functions have been ported to Python.

#     References
#     ----------
#     .. [2] Zhang et al. (2012) NODDI: Practical in vivo neurite orientation
#            dispersion and density imaging of the human brain. NeuroImage, 61: 1000-16
#     """
#     def __init__( self ):
#         self.id         = "NODDI"
#         self.name       = "NODDI"
#         self.maps_name  = [ 'ICVF', 'OD', 'ISOVF' ]
#         self.maps_descr = [ 'Intra-cellular volume fraction', 'Orientation dispersion', 'Isotropic volume fraction' ]

#         self.dPar      = 1.7E-3
#         self.dIso      = 3.0E-3
#         self.IC_VFs    = np.linspace(0.1,0.99,12)
#         self.IC_ODs    = np.hstack((np.array([0.03, 0.06]),np.linspace(0.09,0.99,10)))
#         self.isExvivo  = False


#     def set( self, dPar, dIso, IC_VFs, IC_ODs, isExvivo ):
#         self.dPar      = dPar
#         self.dIso      = dIso
#         self.IC_VFs    = np.array( IC_VFs )
#         self.IC_ODs    = np.array( IC_ODs )
#         self.isExvivo  = isExvivo
#         if isExvivo:
#             self.maps_name  = [ 'ICVF', 'OD', 'ISOVF', 'dot' ]
#             self.maps_descr = [ 'Intra-cellular volume fraction', 'Orientation dispersion', 'Isotropic volume fraction', 'Dot volume fraction' ]
#         else:
#             self.maps_name  = [ 'ICVF', 'OD', 'ISOVF']
#             self.maps_descr = [ 'Intra-cellular volume fraction', 'Orientation dispersion', 'Isotropic volume fraction']


#     def get_params( self ) :
#         params = {}
#         params['id'] = self.id
#         params['name'] = self.name
#         params['dPar'] = self.dPar
#         params['dIso'] = self.dIso
#         params['IC_VFs'] = self.IC_VFs
#         params['IC_ODs'] = self.IC_ODs
#         params['isExvivo'] = self.isExvivo
#         return params


#     def set_solver( self, lambda1 = 5e-1, lambda2 = 1e-3 ):
#         params = {}
#         params['mode']    = 2
#         params['pos']     = True
#         params['lambda1'] = lambda1
#         params['lambda2'] = lambda2
#         return params


#     def generate( self, out_path, aux, idx_in, idx_out, ndirs ):
#         scheme_high = amico.lut.create_high_resolution_scheme( self.scheme )

#         noddi_ic = NODDIIntraCellular(scheme_high)
#         noddi_ec = NODDIExtraCellular(scheme_high)
#         noddi_iso = NODDIIsotropic(scheme_high)

#         nATOMS = len(self.IC_ODs)*len(self.IC_VFs) + 1
#         idx = 0
#         with tqdm(total=nATOMS, ncols=70, bar_format='   |{bar}| {percentage:4.1f}%', disable=(get_verbose()<3)) as progress:
#             # Coupled contributions
#             IC_KAPPAs = 1 / np.tan(self.IC_ODs*np.pi/2)
#             for kappa in IC_KAPPAs:
#                 signal_ic = noddi_ic.get_signal(self.dPar, kappa)
#                 for v_ic in self.IC_VFs:
#                     signal_ec = noddi_ec.get_signal(self.dPar, kappa, v_ic)
#                     signal = v_ic*signal_ic + (1-v_ic)*signal_ec
#                     lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, False, ndirs )
#                     np.save( pjoin( out_path, f'A_{idx+1:03d}.npy') , lm )
#                     idx += 1
#                     progress.update()
#             # Isotropic
#             signal = noddi_iso.get_signal(self.dIso)
#             lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, True, ndirs )
#             np.save( pjoin( out_path, f'A_{nATOMS:03d}.npy') , lm )
#             progress.update()


#     def resample( self, in_path, idx_out, Ylm_out, doMergeB0, ndirs ):
#         nATOMS = len(self.IC_ODs)*len(self.IC_VFs) + 1
#         if doMergeB0:
#             nS = 1+self.scheme.dwi_count
#             merge_idx = np.hstack((self.scheme.b0_idx[0],self.scheme.dwi_idx))
#         else:
#             nS = self.scheme.nS
#             merge_idx = np.arange(nS)
#         KERNELS = {}
#         KERNELS['model'] = self.id
#         KERNELS['wm']    = np.zeros( (nATOMS-1,ndirs,nS), dtype=np.float32 )
#         KERNELS['iso']   = np.zeros( nS, dtype=np.float32 )
#         KERNELS['kappa'] = np.zeros( nATOMS-1, dtype=np.float32 )
#         KERNELS['icvf']  = np.zeros( nATOMS-1, dtype=np.float32 )
#         KERNELS['norms'] = np.zeros( (self.scheme.dwi_count, nATOMS-1) )

#         idx = 0
#         with tqdm(total=nATOMS, ncols=70, bar_format='   |{bar}| {percentage:4.1f}%', disable=(get_verbose()<3)) as progress:
#             # Coupled contributions
#             for i in range( len(self.IC_ODs) ):
#                 for j in range( len(self.IC_VFs) ):
#                     lm = np.load( pjoin( in_path, f'A_{idx+1:03d}.npy' ) )
#                     if lm.shape[0] != ndirs:
#                         ERROR( 'Outdated LUT. Call "generate_kernels( regenerate=True )" to update the LUT' )
#                     KERNELS['wm'][idx,:,:] = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, False, ndirs )[:,merge_idx]
#                     KERNELS['kappa'][idx] = 1.0 / np.tan( self.IC_ODs[i]*np.pi/2.0 )
#                     KERNELS['icvf'][idx]  = self.IC_VFs[j]
#                     if doMergeB0:
#                         KERNELS['norms'][:,idx] = 1 / np.linalg.norm( KERNELS['wm'][idx,0,1:] ) # norm of coupled atoms (for l1 minimization)
#                     else:
#                         KERNELS['norms'][:,idx] = 1 / np.linalg.norm( KERNELS['wm'][idx,0,self.scheme.dwi_idx] ) # norm of coupled atoms (for l1 minimization)
#                     idx += 1
#                     progress.update()
#             # Isotropic
#             lm = np.load( pjoin( in_path, f'A_{nATOMS:03d}.npy' ) )
#             KERNELS['iso'] = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, True, ndirs )[merge_idx]
#             progress.update()

#         return KERNELS


#     def fit( self, y, dirs, KERNELS, params, htable ) :
#         singleb0 = True if len(y) == (1+self.scheme.dwi_count) else False
#         nD = dirs.shape[0]
#         if nD != 1 :
#             ERROR( '"%s" model requires exactly 1 orientation' % self.name )

#         # prepare DICTIONARY from dir and lookup tables
#         nWM = len(self.IC_ODs)*len(self.IC_VFs)
#         nATOMS = nWM + 1
#         if self.isExvivo == True :
#             nATOMS += 1
#         lut_idx = amico.lut.dir_TO_lut_idx( dirs[0], htable )
#         A = np.ones( (len(y), nATOMS), dtype=np.float64, order='F' )
#         A[:,:nWM] = KERNELS['wm'][:,lut_idx,:].T
#         A[:,-1]  = KERNELS['iso']


#         # estimate CSF partial volume (and isotropic restriction, if exvivo) and remove from signal
#         x, _ = scipy.optimize.nnls( A, y )
#         yy = y - x[-1]*A[:,-1]
#         if self.isExvivo == True :
#             yy = yy - x[-2]*A[:,-2]
#         yy[ yy<0 ] = 0

#         # estimate IC and EC compartments and promote sparsity
#         if singleb0:
#             An = A[1:, :nWM] * KERNELS['norms']
#             yy = yy[1:].reshape(-1,1)
#         else:
#             An = A[ self.scheme.dwi_idx, :nWM ] * KERNELS['norms']
#             yy = yy[ self.scheme.dwi_idx ].reshape(-1,1)
#         x = spams.lasso( np.asfortranarray(yy), D=np.asfortranarray(An), numThreads=1, **params ).todense().A1

#         # debias coefficients
#         x = np.append( x, 1 )
#         if self.isExvivo == True :
#             x = np.append( x, 1 )
#         idx = x>0
#         x[idx], _ = scipy.optimize.nnls( A[:,idx], y )

#         # return estimates
#         xx = x / ( x.sum() + 1e-16 )
#         xWM  = xx[:nWM]
#         fISO = xx[-1]
#         xWM = xWM / ( xWM.sum() + 1e-16 )
#         f1 = np.dot( KERNELS['icvf'], xWM )
#         f2 = np.dot( (1.0-KERNELS['icvf']), xWM )
#         v = f1 / ( f1 + f2 + 1e-16 )
#         k = np.dot( KERNELS['kappa'], xWM )
#         od = 2.0/np.pi * np.arctan2(1.0,k)

#         if self.isExvivo:
#             return [v, od, fISO, xx[-2]], dirs, x, A
#         else:
#             return [v, od, fISO], dirs, x, A



class FreeWater( BaseModel ) :
    """Implements the Free-Water model.
    """
    def __init__( self ) :
        self.id         = 'FreeWater'
        self.name       = 'Free-Water'
        self.type       = 'Human'

        if self.type == 'Mouse' :
            self.maps_name  = [ 'FiberVolume', 'FW', 'FW_blood', 'FW_csf' ]
            self.maps_descr = [ 'fiber volume fraction',
                                'Isotropic free-water volume fraction',
                                'FW blood', 'FW csf' ]

            # for mouse imaging
            self.d_par   = 1.0E-3
            self.d_perps = np.linspace(0.15,0.55,10)*1E-3
            self.d_isos  = [1.5e-3, 3e-3]

        else :
            self.maps_name  = [ 'FiberVolume', 'FW' ]
            self.maps_descr = [ 'fiber volume fraction',
                                'Isotropic free-water volume fraction']
            self.d_par   = 1.0E-3                       # Parallel diffusivity [mm^2/s]
            self.d_perps = np.linspace(0.1,1.0,10)*1E-3 # Parallel diffusivities [mm^2/s]
            self.d_isos  = [ 2.5E-3 ]                   # Isotropic diffusivities [mm^2/s]


    def set( self, d_par, d_perps, d_isos, type ) :
        self.d_par   = d_par
        self.d_perps = d_perps
        self.d_isos  = d_isos
        self.type    = type

        if self.type == 'Mouse' :
            self.maps_name  = [ 'FiberVolume', 'FW', 'FW_blood', 'FW_csf' ]
            self.maps_descr = [ 'fiber volume fraction',
                                'Isotropic free-water volume fraction',
                                'FW blood', 'FW csf' ]

        else :
            self.maps_name  = [ 'FiberVolume', 'FW' ]
            self.maps_descr = [ 'fiber volume fraction',
                                'Isotropic free-water volume fraction']

        PRINT('      %s settings for Freewater elimination... ' % self.type)
        PRINT('             -iso  compartments: ', self.d_isos)
        PRINT('             -perp compartments: ', self.d_perps)
        PRINT('             -para compartments: ', self.d_par)


    def get_params( self ) :
        params = {}
        params['id'] = self.id
        params['name'] = self.name
        params['d_par'] = self.d_par
        params['d_perps'] = self.d_perps
        params['d_isos'] = self.d_isos
        params['type'] = self.type
        return params


    def set_solver( self, lambda1 = 0.0, lambda2 = 1e-3 ):
        params = {}
        params['mode']    = 2
        params['pos']     = True
        params['lambda1'] = lambda1
        params['lambda2'] = lambda2

        # need more regul for mouse data
        if self.type == 'Mouse' :
            lambda2 = 0.25

        return params


    def generate( self, out_path, aux, idx_in, idx_out, ndirs ) :
        scheme_high = amico.lut.create_high_resolution_scheme( self.scheme )

        zeppelin = Zeppelin(scheme_high)
        ball = Ball(scheme_high)

        nATOMS = len(self.d_perps) + len(self.d_isos)
        idx = 0
        with tqdm(total=nATOMS, ncols=70, bar_format='   |{bar}| {percentage:4.1f}%', disable=(get_verbose()<3)) as progress:
            # Tensor compartment(s)
            for d in self.d_perps :
                signal = zeppelin.get_signal(self.d_par, d)
                lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, False, ndirs )
                np.save( pjoin( out_path, f'A_{idx+1:03d}.npy' ), lm )
                idx += 1
                progress.update()

            # Isotropic compartment(s)
            for d in self.d_isos :
                signal = ball.get_signal(d)
                lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, True, ndirs )
                np.save( pjoin( out_path, f'A_{idx+1:03d}.npy' ), lm )
                idx += 1
                progress.update()


    def resample( self, in_path, idx_out, Ylm_out, doMergeB0, ndirs ) :
        if doMergeB0:
            nS = 1+self.scheme.dwi_count
            merge_idx = np.hstack((self.scheme.b0_idx[0],self.scheme.dwi_idx))
        else:
            nS = self.scheme.nS
            merge_idx = np.arange(nS)
        KERNELS = {}
        KERNELS['model'] = self.id
        KERNELS['D']     = np.zeros( (len(self.d_perps),ndirs,nS), dtype=np.float32 )
        KERNELS['CSF']   = np.zeros( (len(self.d_isos),nS), dtype=np.float32 )

        nATOMS = len(self.d_perps) + len(self.d_isos)
        idx = 0
        with tqdm(total=nATOMS, ncols=70, bar_format='   |{bar}| {percentage:4.1f}%', disable=(get_verbose()<3)) as progress:
            # Tensor compartment(s)
            for i in range(len(self.d_perps)) :
                lm = np.load( pjoin( in_path, f'A_{idx+1:03d}.npy' ) )
                if lm.shape[0] != ndirs:
                    ERROR( 'Outdated LUT. Call "generate_kernels( regenerate=True )" to update the LUT' )
                KERNELS['D'][i,...] = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, False, ndirs )[:,merge_idx]
                idx += 1
                progress.update()

            # Isotropic compartment(s)
            for i in range(len(self.d_isos)) :
                lm = np.load( pjoin( in_path, f'A_{idx+1:03d}.npy' ) )
                KERNELS['CSF'][i,...] = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, True, ndirs )[merge_idx]
                idx += 1
                progress.update()

        return KERNELS


    def fit( self, y, dirs, KERNELS, params, htable ) :
        nD = dirs.shape[0]
        if nD > 1 : # model works only with one direction
            ERROR( '"%s" model requires exactly 1 orientation' % self.name )

        n1 = len(self.d_perps)
        n2 = len(self.d_isos)
        nATOMS = n1+n2
        if nATOMS == 0 : # empty dictionary
            return [0, 0], None, None, None

        # prepare DICTIONARY from dir and lookup tables
        lut_idx = amico.lut.dir_TO_lut_idx( dirs[0], htable )
        A = np.zeros( (len(y), nATOMS), dtype=np.float64, order='F' )
        A[:,:(nD*n1)] = KERNELS['D'][:,lut_idx,:].T
        A[:,(nD*n1):] = KERNELS['CSF'].T

        # fit
        x = spams.lasso( np.asfortranarray( y.reshape(-1,1) ), D=A, numThreads=1, **params ).todense().A1

        # return estimates
        v = x[ :n1 ].sum() / ( x.sum() + 1e-16 )

        # checking that there is more than 1 isotropic compartment
        if self.type == 'Mouse' :
            v_blood = x[ n1 ] / ( x.sum() + 1e-16 )
            v_csf = x[ n1+1 ] / ( x.sum() + 1e-16 )

            return [ v, 1-v, v_blood, v_csf ], dirs, x, A

        else :
            return [ v, 1-v ], dirs, x, A



class VolumeFractions( BaseModel ) :
    """Implements a simple model where each compartment contributes only with
       its own volume fraction. This model has been created to test there
       ability to remove false positive fibers with COMMIT.
    """

    def __init__( self ) :
        self.id         = 'VolumeFractions'
        self.name       = 'Volume fractions'
        self.maps_name  = [ ]
        self.maps_descr = [ ]


    def get_params( self ) :
        params = {}
        params['id'] = self.id
        params['name'] = self.name
        return params


    def set_solver( self ) :
        ERROR( 'Not implemented' )


    def generate( self, out_path, aux, idx_in, idx_out, ndirs ) :
        return


    def resample( self, in_path, idx_out, Ylm_out, doMergeB0, ndirs ) :
        if doMergeB0:
            nS = 1+self.scheme.dwi_count
            merge_idx = np.hstack((self.scheme.b0_idx[0],self.scheme.dwi_idx))
        else:
            nS = self.scheme.nS
            merge_idx = np.arange(nS)

        KERNELS = {}
        KERNELS['model'] = self.id
        KERNELS['wmr']   = np.ones( (1,ndirs,nS), dtype=np.float32 )
        KERNELS['wmh']   = np.ones( (0,ndirs,nS), dtype=np.float32 )
        KERNELS['iso']   = np.ones( (0,nS), dtype=np.float32 )

        return KERNELS


    def fit( self, y, dirs, KERNELS, params ) :
        ERROR( 'Not implemented' )



class SANDI( BaseModel ) :
    """Implements the SANDI model [1].

    The intra-cellular contributions from within the neural cells are modeled as intra-soma + intra-neurite,
    with the soma modelled as "sphere" of radius (Rs) and fixed intra-soma diffusivity (d_is) to 3 micron^2/ms;
    the neurites are modelled as randomly oriented sticks with axial intra-neurite diffusivity (d_in).
    Extra-cellular contributions are modeled as isotropic gaussian diffusion, i.e. "ball", with the mean diffusivity (d_iso)

    NB: this model works only with direction-averaged signal and schemes containing the full specification of
        the diffusion gradients (eg gradient strength, small delta etc).

    References
    ----------
    .. [1] Palombo, Marco, et al. "SANDI: a compartment-based model for non-invasive apparent soma and neurite imaging by diffusion MRI." Neuroimage 215 (2020): 116835.
    """

    def __init__( self ) :
        self.id         = 'SANDI'
        self.name       = 'SANDI'
        self.maps_name  = [ 'fsoma', 'fneurite', 'fextra', 'Rsoma', 'Din', 'De' ]
        self.maps_descr = [ 'Intra-soma volume fraction', 'Intra-neurite volume fraction', 'Extra-cellular volume fraction', 'Apparent soma radius', 'Neurite axial diffusivity', 'Extra-cellular mean diffusivity' ]

        self.d_is   = 3.0E-3                         # Intra-soma diffusivity [mm^2/s]
        self.Rs     = np.linspace(1.0,12.0,5) * 1E-6 # Radii of the soma [meters]
        self.d_in   = np.linspace(0.25,3.0,5) * 1E-3 # Intra-neurite diffusivitie(s) [mm^2/s]
        self.d_isos = np.linspace(0.25,3.0,5) * 1E-3 # Extra-cellular isotropic mean diffusivitie(s) [mm^2/s]


    def set( self, d_is, Rs, d_in, d_isos ) :
        self.d_is   = d_is
        self.Rs     = np.array(Rs)
        self.d_in   = np.array(d_in)
        self.d_isos = np.array(d_isos)


    def get_params( self ) :
        params = {}
        params['id']     = self.id
        params['name']   = self.name
        params['d_is']   = self.d_is
        params['Rs']     = self.Rs
        params['d_in']   = self.d_in
        params['d_isos'] = self.d_isos
        return params


    def set_solver( self, lambda1 = 0.0, lambda2 = 5.0E-3 ) :
        params = {}
        params['mode']    = 2
        params['pos']     = True
        params['lambda1'] = lambda1
        params['lambda2'] = lambda2
        return params


    def generate( self, out_path, aux, idx_in, idx_out, ndirs ) :
        if self.scheme.version != 1 :
            ERROR( 'This model requires a "VERSION: STEJSKALTANNER" scheme' )

        scheme_high = amico.lut.create_high_resolution_scheme( self.scheme )

        sphere = SphereGPD(scheme_high)
        astrosticks = Astrosticks(scheme_high)
        ball = Ball(scheme_high)

        nATOMS = len(self.Rs) + len(self.d_in) + len(self.d_isos)
        idx = 0
        with tqdm(total=nATOMS, ncols=70, bar_format='   |{bar}| {percentage:4.1f}%', disable=(get_verbose()<3)) as progress:
            # Soma = SPHERE
            for R in self.Rs :
                signal = sphere.get_signal(self.d_is, R)
                lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, True, ndirs )
                np.save( pjoin( out_path, f'A_{idx+1:03d}.npy' ), lm )
                idx += 1
                progress.update()
            # Neurites = ASTRO STICKS
            for d in self.d_in :
                signal = astrosticks.get_signal(d)
                lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, True, ndirs )
                np.save( pjoin( out_path, f'A_{idx+1:03d}.npy' ), lm )
                idx += 1
                progress.update()
            # Extra-cellular = BALL
            for d in self.d_isos :
                signal = ball.get_signal(d)
                lm = amico.lut.rotate_kernel( signal, aux, idx_in, idx_out, True, ndirs )
                np.save( pjoin( out_path, f'A_{idx+1:03d}.npy' ), lm )
                idx += 1
                progress.update()


    def resample( self, in_path, idx_out, Ylm_out, doMergeB0, ndirs ) :
        nATOMS = len(self.Rs) + len(self.d_in) + len(self.d_isos)
        if doMergeB0:
            nS = 1+self.scheme.dwi_count
            merge_idx = np.hstack((self.scheme.b0_idx[0],self.scheme.dwi_idx))
        else:
            nS = self.scheme.nS
            merge_idx = np.arange(nS)
        KERNELS = {}
        KERNELS['model']  = self.id
        KERNELS['signal'] = np.zeros( (nS,nATOMS), dtype=np.float64, order='F' )
        KERNELS['norms']  = np.zeros( nATOMS, dtype=np.float64 )

        idx = 0
        with tqdm(total=nATOMS, ncols=70, bar_format='   |{bar}| {percentage:4.1f}%', disable=(get_verbose()<3)) as progress:
            # Soma = SPHERE
            for i in range(len(self.Rs)) :
                lm = np.load( pjoin( in_path, f'A_{idx+1:03d}.npy' ) )
                signal = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, True, ndirs )[merge_idx].T
                KERNELS['norms'][idx] = 1.0 / np.linalg.norm( signal )
                KERNELS['signal'][:,idx] = signal * KERNELS['norms'][idx]
                idx += 1
                progress.update()
            # Neurites = STICKS
            for i in range(len(self.d_in)) :
                lm = np.load( pjoin( in_path, f'A_{idx+1:03d}.npy' ) )
                signal = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, True, ndirs )[merge_idx].T
                KERNELS['norms'][idx] = 1.0 / np.linalg.norm( signal )
                KERNELS['signal'][:,idx] = signal * KERNELS['norms'][idx]
                idx += 1
                progress.update()
            # Extra-cellular = BALL
            for i in range(len(self.d_isos)) :
                lm = np.load( pjoin( in_path, f'A_{idx+1:03d}.npy' ) )
                signal = amico.lut.resample_kernel( lm, self.scheme.nS, idx_out, Ylm_out, True, ndirs )[merge_idx].T
                KERNELS['norms'][idx] = 1.0 / np.linalg.norm( signal )
                KERNELS['signal'][:,idx] = signal * KERNELS['norms'][idx]
                idx += 1
                progress.update()

        return KERNELS


    def fit( self, y, dirs, KERNELS, params, htable ) :
        # if dictionary is empty
        if KERNELS['signal'].shape[1] == 0 :
            return [0, 0, 0, 0, 0, 0], None, None, None

        # fit
        x = spams.lasso( np.asfortranarray( y.reshape(-1,1) ), D=KERNELS['signal'], numThreads=1, **params ).todense().A1
        x = x*KERNELS['norms']

        # return estimates
        n1 = len(self.Rs)
        n2 = len(self.d_in)
        xsph = x[:n1]
        xstk = x[n1:n1+n2]
        xiso = x[n1+n2:]

        fsoma    = xsph.sum()/(x.sum()+1e-16)
        fneurite = xstk.sum()/(x.sum()+1e-16)
        fextra   = xiso.sum()/(x.sum()+1e-16)
        Rsoma    = 1E6*np.dot(self.Rs,xsph)/(xsph.sum()+1e-16 )     # Sphere radius [micron]
        Din      = 1E3*np.dot(self.d_in,xstk)/(xstk.sum()+1e-16 )   # Intra-stick diffusivity [micron^2/ms]
        De       = 1E3*np.dot(self.d_isos,xiso)/(xiso.sum()+1e-16 ) # Extra-cellular diffusivity [micron^2/ms]

        return [fsoma, fneurite, fextra, Rsoma, Din, De], dirs, x, KERNELS['signal']
