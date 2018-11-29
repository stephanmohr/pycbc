# Copyright (C) 2017 Collin Capano, Chris Biwer
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
This modules provides spin distributions of CBCs.
"""

import numpy
from pycbc import conversions
from pycbc.distributions.uniform import Uniform
from pycbc.distributions.angular import UniformAngle
from pycbc.distributions.power_law import UniformPowerLaw, TruncatedPowerLaw 
from pycbc.distributions.arbitrary import Arbitrary
from pycbc.distributions.bounded import get_param_bounds_from_config, \
    VARARGS_DELIM, BoundedDist

class IndependentChiPChiEff(Arbitrary):
    r"""A distribution such that :math:`\chi_{\mathrm{eff}}` and
    :math:`\chi_p` are uniform and independent of each other.

    To ensure constraints are applied correctly, this distribution produces all
    three components of both spins as well as the component masses.

    Parameters
    ----------
    mass1 : BoundedDist, Bounds, or tuple
        The distribution or bounds to use for mass1. Must be either a
        BoundedDist giving the distribution on mass1, or bounds (as
        either a Bounds instance or a tuple) giving the minimum and maximum
        values to use for mass1. If the latter, a Uniform distribution will
        be used.
    mass2 : BoundedDist, Bounds, or tuple
        The distribution or bounds to use for mass2. Syntax is the same as
        mass1.
    chi_eff : BoundedDist, Bounds, or tuple; optional
        The distribution or bounds to use for :math:`chi_eff`. Syntax is the
        same as mass1, except that None may also be passed. In that case,
        `(-1, 1)` will be used for the bounds. Default is None.
    chi_a : BoundedDist, Bounds, or tuple; optional
        The distribution or bounds to use for :math:`chi_a`. Syntax is the
        same as mass1, except that None may also be passed. In that case,
        `(-1, 1)` will be used for the bounds. Default is None.
    xi_bounds : Bounds or tuple, optional
        The bounds to use for :math:`\xi_1` and :math:`\xi_2`. Must be
        :math:`\in (0, 1)`. If None (the default), will be `(0, 1)`.
    nsamples : int, optional
        The number of samples to use for the internal kde. The larger the
        number of samples, the more accurate the pdf will be, but the longer
        it will take to evaluate. Default is 10000.
    seed : int, optional
        Seed value to use for the number generator for the kde. The current
        random state of numpy will be saved prior to setting the seed. After
        the samples are generated, the state will be set back to what it was.
        If None provided, will use 0.
    """
    name = "independent_chip_chieff"
    _params = ['mass1', 'mass2', 'xi1', 'xi2', 'chi_eff', 'chi_a',
               'phi_a', 'phi_s']

    def __init__(self, mass1=None, mass2=None, chi_eff=None, chi_a=None,
                 xi_bounds=None, nsamples=None, seed=None):

        if isinstance(mass1, BoundedDist):
            self.mass1_distr = mass1
        else:
            self.mass1_distr = Uniform(mass1=mass1)
        if isinstance(mass2, BoundedDist):
            self.mass2_distr = mass2
        else:
            self.mass2_distr = Uniform(mass2=mass2)
        # chi eff
        if isinstance(chi_eff, BoundedDist):
            self.chieff_distr = chi_eff
        else:
            if chi_eff is None:
                chi_eff = (-1., 1.)
            self.chieff_distr = Uniform(chi_eff=chi_eff)
        if isinstance(chi_a, BoundedDist):
            self.chia_distr = chi_a
        else:
            if chi_a is None:
                chi_a = (-1., 1.)
            self.chia_distr = Uniform(chi_a=chi_a)
        # xis
        if xi_bounds is None:
            xi_bounds = (0, 1.)
        if (xi_bounds[0] > 1. or xi_bounds[0] < 0.) or (
            xi_bounds[1] > 1. or xi_bounds[1] < 0.):
            raise ValueError("xi bounds must be in [0, 1)")
        self.xi1_distr = UniformPowerLaw(dim=0.5, xi1=xi_bounds)
        self.xi2_distr = UniformPowerLaw(dim=0.5, xi2=xi_bounds)
        # the angles
        self.phia_distr = UniformAngle(phi_a=(0,2))
        self.phis_distr = UniformAngle(phi_s=(0,2))
        self.distributions = {'mass1': self.mass1_distr,
                              'mass2': self.mass2_distr,
                              'xi1': self.xi1_distr,
                              'xi2': self.xi2_distr,
                              'chi_eff': self.chieff_distr,
                              'chi_a': self.chia_distr,
                              'phi_a': self.phia_distr,
                              'phi_s': self.phis_distr}
        # create random variables for the kde
        if nsamples is None:
            nsamples = 1e4
        # save the current random state
        rstate = numpy.random.get_state()
        # set the seed
        if seed is None:
            seed = 0
        numpy.random.seed(seed)
        rvals = self.rvs(size=int(nsamples))
        # reset the random state back to what it was
        numpy.random.set_state(rstate)
        bounds = dict(b for distr in self.distributions.values()
                        for b in distr.bounds.items())
        super(IndependentChiPChiEff, self).__init__(mass1=rvals['mass1'],
            mass2=rvals['mass2'], xi1=rvals['xi1'], xi2=rvals['xi2'],
            chi_eff=rvals['chi_eff'], chi_a=rvals['chi_a'],
            phi_a=rvals['phi_a'], phi_s=rvals['phi_s'],
            bounds=bounds)

    def _constraints(self, values):
        """Applies physical constraints to the given parameter values.

        Parameters
        ----------
        values : {arr or dict}
            A dictionary or structured array giving the values.

        Returns
        -------
        bool
            Whether or not the values satisfy physical
        """
        mass1, mass2, phi_a, phi_s, chi_eff, chi_a, xi1, xi2, _ = \
            conversions.ensurearray(values['mass1'], values['mass2'],
                                    values['phi_a'], values['phi_s'],
                                    values['chi_eff'], values['chi_a'],
                                    values['xi1'], values['xi2'])
        s1x = conversions.spin1x_from_xi1_phi_a_phi_s(xi1, phi_a, phi_s)
        s2x = conversions.spin2x_from_mass1_mass2_xi2_phi_a_phi_s(mass1, mass2,
            xi2, phi_a, phi_s)
        s1y = conversions.spin1y_from_xi1_phi_a_phi_s(xi1, phi_a, phi_s)
        s2y = conversions.spin2y_from_mass1_mass2_xi2_phi_a_phi_s(mass1, mass2,
            xi2, phi_a, phi_s)
        s1z = conversions.spin1z_from_mass1_mass2_chi_eff_chi_a(mass1, mass2,
            chi_eff, chi_a)
        s2z = conversions.spin2z_from_mass1_mass2_chi_eff_chi_a(mass1, mass2,
            chi_eff, chi_a)
        test = ((s1x**2. + s1y**2. + s1z**2.) < 1.) & \
               ((s2x**2. + s2y**2. + s2z**2.) < 1.)
        return test

    def __contains__(self, params):
        """Determines whether the given values are in each parameter's bounds
        and satisfy the constraints.
        """
        isin = all([params in dist for dist in self.distributions.values()])
        if not isin:
            return False
        # in the individual distributions, apply constrains
        return self._constraints(params)

    def _draw(self, size=1, **kwargs):
        """Draws random samples without applying physical constrains.
        """
        # draw masses
        try:
            mass1 = kwargs['mass1']
        except KeyError:
            mass1 = self.mass1_distr.rvs(size=size)['mass1']
        try:
            mass2 = kwargs['mass2']
        except KeyError:
            mass2 = self.mass2_distr.rvs(size=size)['mass2']
        # draw angles
        try:
            phi_a = kwargs['phi_a']
        except KeyError:
            phi_a = self.phia_distr.rvs(size=size)['phi_a']
        try:
            phi_s = kwargs['phi_s']
        except KeyError:
            phi_s = self.phis_distr.rvs(size=size)['phi_s']
        # draw chi_eff, chi_a
        try:
            chi_eff = kwargs['chi_eff']
        except KeyError:
            chi_eff = self.chieff_distr.rvs(size=size)['chi_eff']
        try:
            chi_a = kwargs['chi_a']
        except KeyError:
            chi_a = self.chia_distr.rvs(size=size)['chi_a']
        # draw xis
        try:
            xi1 = kwargs['xi1']
        except KeyError:
            xi1 = self.xi1_distr.rvs(size=size)['xi1']
        try:
            xi2 = kwargs['xi2']
        except KeyError:
            xi2 = self.xi2_distr.rvs(size=size)['xi2']
        dtype = [(p, float) for p in self.params]
        arr = numpy.zeros(size, dtype=dtype)
        arr['mass1'] = mass1
        arr['mass2'] = mass2
        arr['phi_a'] = phi_a
        arr['phi_s'] = phi_s
        arr['chi_eff'] = chi_eff
        arr['chi_a'] = chi_a
        arr['xi1'] = xi1
        arr['xi2'] = xi2
        return arr

    def apply_boundary_conditions(self, **kwargs):
        return kwargs


    def rvs(self, size=1, **kwargs):
        """Returns random values for all of the parameters.
        """
        size = int(size)
        dtype = [(p, float) for p in self.params]
        arr = numpy.zeros(size, dtype=dtype)
        remaining = size
        keepidx = 0
        while remaining:
            draws = self._draw(size=remaining, **kwargs)
            mask = self._constraints(draws)
            addpts = mask.sum()
            arr[keepidx:keepidx+addpts] = draws[mask]
            keepidx += addpts
            remaining = size - keepidx
        return arr


    @classmethod
    def from_config(cls, cp, section, variable_args):
        """Returns a distribution based on a configuration file. The parameters
        for the distribution are retrieved from the section titled
        "[`section`-`variable_params`]" in the config file.

        Parameters
        ----------
        cp : pycbc.workflow.WorkflowConfigParser
            A parsed configuration file that contains the distribution
            options.
        section : str
            Name of the section in the configuration file.
        variable_args : str
            The names of the parameters for this distribution, separated by
            `prior.VARARGS_DELIM`. These must appear in the "tag" part
            of the section header.

        Returns
        -------
        IndependentChiPChiEff
            A distribution instance.
        """
        tag = variable_args
        variable_args = variable_args.split(VARARGS_DELIM)
        if not set(variable_args) == set(cls._params):
            raise ValueError("Not all parameters used by this distribution "
                             "included in tag portion of section name")
        # get the bounds for the setable parameters
        mass1 = get_param_bounds_from_config(cp, section, tag, 'mass1')
        mass2 = get_param_bounds_from_config(cp, section, tag, 'mass2')
        chi_eff = get_param_bounds_from_config(cp, section, tag, 'chi_eff')
        chi_a = get_param_bounds_from_config(cp, section, tag, 'chi_a')
        xi_bounds = get_param_bounds_from_config(cp, section, tag, 'xi_bounds')
        if cp.has_option('-'.join([section, tag]), 'nsamples'):
            nsamples = int(cp.get('-'.join([section, tag]), 'nsamples'))
        else:
            nsamples = None
        return cls(mass1=mass1, mass2=mass2, chi_eff=chi_eff, chi_a=chi_a,
                   xi_bounds=xi_bounds, nsamples=nsamples)



class UniformChiPChiEffTruncatedPowerLaw(Arbitrary):
    r"""A distribution such that :math:`\chi_{\mathrm{eff}}` and 
    :math:`\chi_p` have uniform densities in the following sense:

    Keeping all other parameters fixed and varying only 
    :math:`\chi_{\mathrm{eff}}` or :math:`\chi_p`, the density will 
    stay constant. This does not imply that the marginals of 
    :math:`\chi_{\mathrm{eff}}` or :math:`\chi_p` are uniform, nor that 
    the two parameters are independent.

    This distribution uses analytically derived distributions for 
    the perpendicular components magnitudes of the spins, zeta1 and zeta2.
    In particular, zeta1 has a truncated power law density. 

    Parameters
    ----------
    mass1 : BoundedDist, Bounds, or tuple 
        The distribution or bounds to use for mass1. Must be either a 
        BoundedDist giving the distribution on mass1, or bounds (as 
        either a Bounds instance or a tuple) giving the minimum and maximum 
        values to use for mass1. If the latter, a Uniform distribution will 
        be used.
    mass2 : BoundedDist, Bounds, or tuple
        The distribution or bounds to use for mass2. Same Syntax as for mass1.
    chi_eff : BoundedDist, Bounds or tuple; optional
        The distribution or bounds to use for :math:`\chi_{eff}`. Syntax is the 
        same as mass1, except that None may also be pased. In that case, 
        `(-1,1)` will be used for the bounds of a uniform marginal.
        Default is None. 
    chi_a > BoundedDist, Bounds, or tuple; optional
        The distribution or bounds to use for :math:`chi_a`. Syntax is the
        same as mass1, except that None may also be passed. In that case,
        `(-1, 1)` will be used for the bounds. Default is None.
    nsamples : int, optional
        The number of samples to use for the internal kde. The larger the 
        number of samples, the more accurate the pdf will be, but the longer 
        it will take to evaluate. Default is 10000.
    seed : int, optional
        Seed value to use for the number generator for the kde. The curret 
        random state of numpy will be saved prior to setting the seed. After 
        the samples are gneerated, the state will be set back to what it was.
        If None provided, will use 0.
    """
    name = "uniform_chip_chieff_truncated"
    _params = ['mass1','mass2','zeta1', 'zeta2','chi_eff','chi_a',
               'phi_1','phi_2']
    
    def __init__(self, mass1=None, mass2=None, chi_eff=None, chi_a=None,
                 nsamples=None, seed=None):
        self.drawn = 0
        self.accepted = 0 
        if isinstance(mass1, BoundedDist):
            self.mass1_distr = mass1 
        else:
            self.mass1_distr = Uniform(mass1=mass1)
        if isinstance(mass2, BoundedDist):
            self.mass2_distr = mass2 
        else:
            self.mass2_distr = Uniform(mass2=mass2) 
        # chi_eff
        if isinstance(chi_eff, BoundedDist):
            self.chieff_distr = chi_eff 
        else:
            if chi_eff is None:
                chi_eff = (-1.,1.) 
            self.chieff_distr = Uniform(chi_eff=chi_eff) 
        if isinstance(chi_a,BoundedDist):
            self.chia_distr = chi_a 
        else:
            if chi_a is None:
                chi_a = (-1.,1.) 
            self.chia_distr = Uniform(chi_a=chi_a)
        # zeta distributions! 
        self.zeta1_distr = TruncatedPowerLaw(dim=0.5, zeta1=(0,1))
        self.zeta2_distr = UniformPowerLaw(dim=0.5, zeta2=(0,1))
        # The angles: Now use phi1 and phi2 instead of phia, phis.
        self.phi1_distr = UniformAngle(phi_1=(0,2))
        self.phi2_distr = UniformAngle(phi_2=(0,2))
        self.distributions = {'mass1': self.mass1_distr,
                              'mass2': self.mass2_distr,
                              'zeta1': self.zeta1_distr,
                              'zeta2': self.zeta2_distr,
                              'chi_eff': self.chieff_distr,
                              'chi_a': self.chia_distr,
                              'phi_1': self.phi1_distr,
                              'phi_2': self.phi2_distr}
        # create random samples for the kde
        if nsamples is None:
            nsamples = 1e4
        # save the current random state
        rstate = numpy.random.get_state()
        # set the seed
        if seed is None:
            seed = 0
        numpy.random.seed(seed)
        rvals = self.rvs(size=int(nsamples))
        # reset the random state to what it was
        # better with with-construct?
        numpy.random.set_state(rstate)
        bounds = dict(b for distr in self.distributions.values()
                        for b in distr.bounds.items())
        super(UniformChiPChiEffTruncatedPowerLaw, self).__init__(
            mass1=rvals['mass1'], mass2=rvals['mass2'], 
            zeta1=rvals['zeta1'], zeta2=rvals['zeta2'],
            chi_eff=rvals['chi_eff'], chi_a=rvals['chi_a'],
            phi_1=rvals['phi_1'], phi_2=rvals['phi_2'],
            bounds=bounds)
    
    def _constraints(self,values):
        """
        Applies physical constraints to the given parameter values,
        namely that normalized spin magnitude is smaller than 1.

        Parameters
        ----------
        values : {arr or dict}
            A dictionary or structured array giving the values.

        Return 
        ------
        bool
            Whether or not the values satisfy the physical constraints.
        """
        mass1, mass2, phi_1, phi_2, chi_eff, chi_a, zeta1, zeta2, _ = \
            conversions.ensurearray(values['mass1'], values['mass2'],
                                    values['phi_1'], values['phi_2'],
                                    values['chi_eff'], values['chi_a'],
                                    values['zeta1'], values['zeta2'])
        s1z = conversions.spin1z_from_mass1_mass2_chi_eff_chi_a(mass1,mass2,
            chi_eff, chi_a)
        s2z = conversions.spin2z_from_mass1_mass2_chi_eff_chi_a(mass1,mass2,
            chi_eff,chi_a)
        test = ((s1z**2. + zeta1**2.) < 1.) & \
               ((s2z**2. + zeta2**2.) < 1.)
        return test 

    def __contains__(self,params):
        """
        Determines whether the given values are in each parameters's bounds
        and satisfy the constraints.
        """
        isin = all([params in dist for dist in self.distributions.values()])
        if not isin:
            return False 
        # After the individual distribution constraints apply the physical ones
        return self._constraints(params)

    def _draw(self, size=1, **kwargs):
        """
        Draws random samples without applying physical constraints.
        """
        # draw masses 
        try: 
            mass1 = kwargs['mass1']
        except KeyError:
            mass1 = self.mass1_distr.rvs(size=size)['mass1']
        try:
            mass2 = kwargs['mass2']
        except KeyError:
            mass2 = self.mass2_distr.rvs(size=size)['mass2']
        # draw angles
        mass1, mass2 = numpy.maximum(mass1,mass2), numpy.minimum(mass1,mass2) 
        try:
            phi_1 = kwargs['phi_1']
        except KeyError:
            phi_1 = self.phi1_distr.rvs(size=size)['phi_1']
        try: 
            phi_2 = kwargs['phi_2']
        except KeyError:
            phi_2 = self.phi2_distr.rvs(size=size)['phi_2']
        # draw chi_eff, chi_a
        try:
            chi_eff = kwargs['chi_eff']
        except KeyError:
            chi_eff = self.chieff_distr.rvs(size=size)['chi_eff']
        try:
            chi_a = kwargs['chi_a']
        except KeyError:
            chi_a = self.chia_distr.rvs(size=size)['chi_a']
        # draw zetas 
        q = mass1 / mass2
        B = (4 + 3*q)/(4*q**2 + 3*q) 
        try:
            zeta1 = kwargs['zeta1']
        except KeyError:
            zeta1 = self.zeta1_distr.rvs(size=size,B=B)['zeta1']
        try: 
            zeta2 = kwargs['zeta2']
        except KeyError:
            zeta2 = self.zeta2_distr.rvs(size=size)['zeta2']
        dtype = [(p, float) for p in self._params]
        arr = numpy.zeros(size, dtype=dtype)
        arr['mass1'] = mass1
        arr['mass2'] = mass2 
        arr['phi_1'] = phi_1
        arr['phi_2'] = phi_2 
        arr['chi_eff'] = chi_eff 
        arr['chi_a'] = chi_a 
        arr['zeta1'] = zeta1 
        arr['zeta2'] = zeta2 
        self.drawn += size 
        return arr 
    
    def apply_boundary_conditions(self, **kwargs): 
        return kwargs 
    
    def rvs(self, size=1, **kwargs): 
        """ 
        Returns random values for all of the parameters 
        """
        size = int(size) 
        dtype = [(p,float) for p in self.params] 
        arr = numpy.zeros(size, dtype=dtype)
        remaining = size 
        keepidx = 0
        while remaining:
            draws = self._draw(size=remaining, **kwargs) 
            mask = self._constraints(draws) 
            addpts = mask.sum() 
            arr[keepidx: keepidx+addpts] = draws[mask]
            keepidx += addpts 
            remaining = size - keepidx
        self.accepted += size 
        return arr

    @classmethod 
    def from_config(cls, cp, section, variable_args): 
        """
        Returns a distribution based on a configuration file. The parameters 
        for the distribution are retrieved from the section titled 
        "[`section`-`variable_params`]" in the config file. 

        Parameters
        ----------
        cp : pycbc.workflow.WorkflowConfigParser 
            A pased configuration file that contains the distribution options.
        section : str 
            Name of the section in the configuration file. 
        variable_args : str 
            The names of the parameters for this distribution, separated by 
            `prior.VARARGS_DELIM`. Theses must appear in the "tag" part 
            of the section header. 
        
        Returns 
        -------
        UniformChiPChiEffTruncatedPowerLaw
            A distribution instance. 
        """
        tag = variable_args
        variable_args = variable_args.split(VARARGS_DELIM)
        if not set(variable_args) == set(cls._params):
            raise ValueError("Not all parameters used by this distribution "
                             "included in tag portion of section 'name'")
        # get the bounds for the setable parameters 
        mass1 = get_param_bounds_from_config(cp, section, tag, 'mass1')
        mass2 = get_param_bounds_from_config(cp, section, tag, 'mass2')
        chi_eff = get_param_bounds_from_config(cp, section, tag, 'chi_eff')
        chi_a = get_param_bounds_from_config(cp, section, tag, 'chi_a')
        if cp.has_option('-'.join([section, tag]), 'nsamples'):
            nsamples = int(cp.get('-'.join([section, tag]), 'nsamples'))
        else:
            nsamples = None
        return cls(mass1=mass1, mass2=mass2, chi_eff=chi_eff, chi_a=chi_a,
                   nsamples=nsamples)


class UniformChiPChiEffGamma(Arbitrary):
    r"""A distribution such that :math:`\chi_{\mathrm{eff}}` and 
    :math:`\chi_p` have uniform densities in the following sense:

    Keeping all other parameters fixed and varying only 
    :math:`\chi_{\mathrm{eff}}` or :math:`\chi_p`, the density will 
    stay constant. This does not imply that the marginals of 
    :math:`\chi_{\mathrm{eff}}` or :math:`\chi_p` are uniform, nor that 
    the two parameters are independent.

    This distribution uses 

    Parameters
    ----------
    mass1 : BoundedDist, Bounds, or tuple 
        The distribution or bounds to use for mass1. Must be either a 
        BoundedDist giving the distribution on mass1, or bounds (as 
        either a Bounds instance or a tuple) giving the minimum and maximum 
        values to use for mass1. If the latter, a Uniform distribution will 
        be used.
    mass2 : BoundedDist, Bounds, or tuple
        The distribution or bounds to use for mass2. Same Syntax as for mass1.
    chi_eff : BoundedDist, Bounds or tuple; optional
        The distribution or bounds to use for :math:`\chi_{eff}`. Syntax is the 
        same as mass1, except that None may also be pased. In that case, 
        `(-1,1)` will be used for the bounds of a uniform marginal.
        Default is None. 
    chi_p : BoundedDist, Bounds, or tuple; optional
        The distribution or bounds to use for :math:`chi_p`. Syntax is the
        same as mass1, except that None may also be passed. In that case,
        `(0, 1)` will be used for the bounds. Default is None.
        No checks are done for negative values in chi_p
    nsamples : int, optional
        The number of samples to use for the internal kde. The larger the 
        number of samples, the more accurate the pdf will be, but the longer 
        it will take to evaluate. Default is 10000.
    seed : int, optional
        Seed value to use for the number generator for the kde. The curret 
        random state of numpy will be saved prior to setting the seed. After 
        the samples are gneerated, the state will be set back to what it was.
        If None provided, will use 0.
    """
    name = "uniform_chip_chieff_gamma"
    _params = ['mass1','mass2','zeta1', 'zeta2','s1z','s2z',
               'phi_1','phi_2','chi_p']
    
    def __init__(self, mass1=None, mass2=None, chi_eff=None, chi_p=None,
                 nsamples=None, seed=None):

        # save the number of samples drawn by _draw 
        # and the number of samples accepted by rvs
        # to check how likely a non-physical value is
        self.drawn = 0
        self.accepted = 0
        
        if isinstance(mass1, BoundedDist):
            self.mass1_distr = mass1 
        else:
            self.mass1_distr = Uniform(mass1=mass1)
        if isinstance(mass2, BoundedDist):
            self.mass2_distr = mass2
        else:
            self.mass2_distr = Uniform(mass2=mass2) 
        # chi_eff and chi_p
        if isinstance(chi_eff, BoundedDist):
            self.chieff_distr = chi_eff 
        else:
            if chi_eff is None:
                chi_eff = (-1.,1.) 
            self.chieff_distr = Uniform(chi_eff=chi_eff)
        if isinstance(chi_p, BoundedDist):
            self.chip_distr = chi_p 
        else:
            if chi_p is None:
                chi_p = (0.,1.) 
            self.chip_distr = Uniform(chi_p=chi_p)
        # additional parameters for the sampling algorithm
        # will not be saved 
        self.gammap_distr = Uniform(gamma_p=(-1,1))
        self.gammaz_distr = Uniform(gamma_z=(0,1))
        # The angles: Now use phi1 and phi2 instead of phia, phis.
        self.phi1_distr = UniformAngle(phi_1=(0,2))
        self.phi2_distr = UniformAngle(phi_2=(0,2))

        # add Arbitrary distributions for zeta1, zeta2 and chi_a later
        self.distributions = {'mass1': self.mass1_distr,
                              'mass2': self.mass2_distr,
                              'phi_1': self.phi1_distr,
                              'phi_2': self.phi2_distr,
                              'chi_p': self.chip_distr}
        # save the current random state
        rstate = numpy.random.get_state()
        if nsamples is None:
            nsamples = 1e4
        # set the seed
        if seed is None:
            seed = 0
        numpy.random.seed(seed)
        rvals = self.rvs(size=int(nsamples))
        # reset the random state to what it was
        # better with with-construct?
        numpy.random.set_state(rstate)
        bounds = dict(b for distr in self.distributions.values()
                        for b in distr.bounds.items())
        bounds['zeta1'] = (0.,1.)
        bounds['zeta2'] = (0.,1. )
        bounds['s1z'] = (-1.,1.)
        bounds['s2z'] = (-1.,1.) 
        super(UniformChiPChiEffGamma, self).__init__(
            bounds=bounds,
            mass1=rvals['mass1'], mass2=rvals['mass2'], 
            zeta1=rvals['zeta1'], zeta2=rvals['zeta2'],
            s1z=rvals['s1z'], s2z=rvals['s2z'],
            phi_1=rvals['phi_1'], phi_2=rvals['phi_2'],
            chi_p=rvals['chi_p'])
        self.s1z_distr   = Arbitrary(s1z=rvals['s1z'],
                                    bounds={'s1z':(-1,1)})
        self.s2z_distr   = Arbitrary(s2z=rvals['s2z'],
                                    bounds={'s2z':(-1,1)}) 
        self.zeta1_distr = Arbitrary(zeta1=rvals['zeta1'],
                                    bounds={'zeta1':(0,1)})
        self.zeta2_distr = Arbitrary(zeta2=rvals['zeta2'],
                                    bounds={'zeta2':(0,1)}) 

    def _constraints(self,values):
        """
        Applies physical constraints to the given parameter values,
        namely that normalized spin magnitude is smaller than 1.

        Parameters
        ----------
        values : {arr or dict}
            A dictionary or structured array giving the values.

        Return 
        ------
        bool
            Whether or not the values satisfy the physical constraints.
        """
        mass1, mass2, phi_1, phi_2, s1z, s2z, zeta1, zeta2, _ = \
            conversions.ensurearray(values['mass1'], values['mass2'],
                                    values['phi_1'], values['phi_2'],
                                    values['s1z'], values['s2z'],
                                    values['zeta1'], values['zeta2'])
        test = ((s1z**2. + zeta1**2.) < 1.) & \
               ((s2z**2. + zeta2**2.) < 1.)

        #if not numpy.all(test): 
        #    print("Constraints not fulfilled.")
        #    print(values[:][numpy.logical_not(test)])
        return test 

    def __contains__(self,params):
        """
        Determines whether the given values are in each parameters's bounds
        and satisfy the constraints.
        """
        isin = all([params in dist for dist in self.distributions.values()])
        if not isin:
            return False 
        # After the individual distribution constraints apply the physical ones
        return self._constraints(params)

    def _draw(self, size=1, **kwargs):
        """
        Draws random samples without applying physical constraints.
        """
        # draw angles
        try:
            phi_1 = kwargs['phi_1']
        except KeyError:
            phi_1 = self.phi1_distr.rvs(size=size)['phi_1']
        try: 
            phi_2 = kwargs['phi_2']
        except KeyError:
            phi_2 = self.phi2_distr.rvs(size=size)['phi_2']
        # draw masses 
        try: 
            mass1 = kwargs['mass1']
        except KeyError:
            mass1 = self.mass1_distr.rvs(size=size)['mass1']
        try:
            mass2 = kwargs['mass2']
        except KeyError:
            mass2 = self.mass2_distr.rvs(size=size)['mass2']
        q = mass1 / mass2 
        primary_is_2 = mass1 < mass2 
        onesided_q = numpy.where(primary_is_2, 1/q, q)
        # draw chi_eff and chi_p
        try:
            chi_eff = kwargs['chi_eff']
        except KeyError:
            chi_eff = self.chieff_distr.rvs(size=size)['chi_eff']
        try:
            chi_p = kwargs['chi_p']
        except KeyError:
            chi_p = self.chip_distr.rvs(size=size)['chi_p']
        # draw gammas 
        gamma_p = self.gammap_distr.rvs(size=size)['gamma_p']
        gamma_z = self.gammaz_distr.rvs(size=size)['gamma_z']
        # set the perpendicular components
        B = (4 + 3*onesided_q) / (4*onesided_q**2 + 3*onesided_q) 
        # If chi_p is larger than B, the primary spin must have perpendicular 
        # component chi_p, the other one may have any.
        # Otherwise assign according to gamma_p. 
        primary_chi_perp = numpy.where(chi_p>B, chi_p, 
                        numpy.where(gamma_p>=0, chi_p, -gamma_p*chi_p))
        secondary_chi_perp = numpy.where(chi_p>B, numpy.abs(gamma_p), 
                        numpy.where(gamma_p>=0, gamma_p*chi_p/B, chi_p/B))
        # Do not check, whether chi_eff is physically possible now. 
        # This can equally well be done later and save as lot of checks here.
        primary_chi_z_bound = numpy.sqrt(1-primary_chi_perp**2)
        primary_chi_z = (2*gamma_z-1)*primary_chi_z_bound 
        secondary_chi_z = (1+onesided_q)*chi_eff - onesided_q*primary_chi_z 
        # assign primary and secondary according to mass relationship. 
        s1z = numpy.where(primary_is_2, secondary_chi_z, primary_chi_z) 
        s2z = numpy.where(primary_is_2, primary_chi_z, secondary_chi_z)
        zeta1 = numpy.where(primary_is_2, secondary_chi_perp, primary_chi_perp)
        zeta2 = numpy.where(primary_is_2, primary_chi_perp, secondary_chi_perp)
        # print for checks: At least zeta1 and zeta2 should be smaller than 1
        if not numpy.all(zeta1<=1):
            print("Zeta1 is larger than 1") 
        if not numpy.all(zeta2<=1):
            print("Zeta2 is larger than 1") 
        # save the resulting spin and mass values
        # do not save the gammas
        dtype = [(p, float) for p in self.params]
        arr = numpy.zeros(size, dtype=dtype)
        arr['mass1'] = mass1
        arr['mass2'] = mass2 
        arr['phi_1'] = phi_1
        arr['phi_2'] = phi_2 
        arr['s1z'] = s1z
        arr['s2z'] = s2z
        arr['zeta1'] = zeta1 
        arr['zeta2'] = zeta2 
        arr['chi_p'] = chi_p 
        self.drawn += size 
        return arr 
    
    def apply_boundary_conditions(self, **kwargs): 
        return kwargs 

    def rvs(self, size=1, **kwargs): 
        """ 
        Returns random values for all of the parameters 
        """
        size = int(size) 
        dtype = [(p,float) for p in self.params] 
        arr = numpy.zeros(size, dtype=dtype)
        remaining = size 
        keepidx = 0
        while remaining:
            draws = self._draw(size=remaining, **kwargs) 
            mask = self._constraints(draws) 
            addpts = mask.sum() 
            arr[keepidx: keepidx+addpts] = draws[mask]
            keepidx += addpts 
            remaining = size - keepidx
        self.accepted =+ size 
        return arr

    @classmethod 
    def from_config(cls, cp, section, variable_args): 
        """
        Returns a distribution based on a configuration file. The parameters 
        for the distribution are retrieved from the section titled 
        "[`section`-`variable_params`]" in the config file. 

        Parameters
        ----------
        cp : pycbc.workflow.WorkflowConfigParser 
            A pased configuration file that contains the distribution options.
        section : str 
            Name of the section in the configuration file. 
        variable_args : str 
            The names of the parameters for this distribution, separated by 
            `prior.VARARGS_DELIM`. Theses must appear in the "tag" part 
            of the section header. 
        
        Returns 
        -------
        UniformChiPChiEffGamma
            A distribution instance. 
        """
        tag = variable_args
        variable_args = variable_args.split(VARARGS_DELIM)
        if not set(variable_args) == set(cls._params):
            raise ValueError("Not all parameters used by this distribution "
                             "included in tag portion of section 'name'")
        # get the bounds for the setable parameters 
        mass1 = get_param_bounds_from_config(cp, section, tag, 'mass1')
        mass2 = get_param_bounds_from_config(cp, section, tag, 'mass2')
        chi_eff = get_param_bounds_from_config(cp, section, tag, 'chi_eff')
        chi_p = get_param_bounds_from_config(cp, section, tag, 'chi_p')
        if cp.has_option('-'.join([section, tag]), 'nsamples'):
            nsamples = int(cp.get('-'.join([section, tag]), 'nsamples'))
        else:
            nsamples = None
        return cls(mass1=mass1, mass2=mass2, chi_eff=chi_eff,
                   chi_p=chi_p, nsamples=nsamples)