# Copyright (C) 2016  Christopher M. Biwer
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


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
"""
This modules provides functions for calculating the autocorrelation function
and length of a data series.
"""

import numpy
from math import isnan

from pycbc.filter.matchedfilter import correlate
from pycbc.types import FrequencySeries, TimeSeries, zeros

def calculate_autocov_function(data, delta_t=1.0, unbiased=False):
    """
    Same as calculate_acf but without the normalization
    """
    # if given a TimeSeries instance then get numpy.array
    if isinstance(data, TimeSeries):
        y = data.numpy()
        delta_t = data.delta_t
    else:
        y = data

    # Zero mean
    y = y - y.mean()
    ny_orig = len(y)

    npad = 1
    while npad < 2*ny_orig:
        npad = npad << 1
    ypad = numpy.zeros(npad)
    ypad[:ny_orig] = y

    # FFT data minus the mean
    fdata = TimeSeries(ypad, delta_t=delta_t).to_frequencyseries()

    # correlate
    # do not need to give the congjugate since correlate function does it
    cdata = FrequencySeries(zeros(len(fdata), dtype=fdata.dtype),
                            delta_f=fdata.delta_f, copy=False)
    correlate(fdata, fdata, cdata)

    # IFFT correlated data to get unnormalized autocovariance time series
    acf = cdata.to_timeseries()
    acf = acf[:ny_orig]

    if unbiased:
        acf /= numpy.arange(len(acf), 0, -1) 
    else:
        acf /= len(acf) 

    if isinstance(data, TimeSeries):
        return TimeSeries(acf, delta_t=delta_t)
    else:
        return acf

def calculate_autocov_2(data, delta_t=1.0, unbiased=False):
    """Calculates the autocorrelation function.
    Uses a standard sum formula
    """
    if isinstance(data, TimeSeries):
        y = data.numpy()
        delta_t = data.delta_t 
    else:
        y = data 
    
    # zero mean
    y = data - data.mean()
    n = len(y) 
    ypad = numpy.zeros(2*n)
    ypad[:n] = y 

    acov = numpy.zeros(n) 
    for k in range(n):
        acov[k] = 1/float(n) * numpy.sum(y * numpy.roll(ypad, -k)[0:n] ) 
    
    return acov 

def plot_acov(data, name, normalize=False):
    """ Plots the autocovariance function for data.
    """
    plt.figure()
    if isinstance(data, numpy.ndarray):
        data = [data] 
    for i, d in enumerate(data):
        acov = calculate_autocov_2(d) 
        if normalize:
            acov /= acov[0]
        plt.plot(acov, label = i)
    plt.axhline(color='black', linewidth=0.7)
    plt.legend()
    plt.savefig(name)
    plt.show()

def calculate_acf(data, delta_t=1.0, unbiased=False):
    r"""Calculates the one-sided autocorrelation function.

    Calculates the autocorrelation function (ACF) and returns the one-sided
    ACF. The ACF is defined as the autocovariance divided by the variance. The
    ACF can be estimated using

    .. math::

        \hat{R}(k) = \frac{1}{n \sigma^{2}} \sum_{t=1}^{n-k} \left( X_{t} - \mu \right) \left( X_{t+k} - \mu \right)

    Where :math:`\hat{R}(k)` is the ACF, :math:`X_{t}` is the data series at
    time t, :math:`\mu` is the mean of :math:`X_{t}`, and :math:`\sigma^{2}` is
    the variance of :math:`X_{t}`.

    Parameters
    -----------
    data : TimeSeries or numpy.array
        A TimeSeries or numpy.array of data.
    delta_t : float
        The time step of the data series if it is not a TimeSeries instance.
    unbiased : bool
        If True the normalization of the autocovariance function is n-k
        instead of n. This is called the unbiased estimation of the
        autocovariance. Note that this does not mean the ACF is unbiased.

    Returns
    -------
    acf : numpy.array
        If data is a TimeSeries then acf will be a TimeSeries of the
        one-sided ACF. Else acf is a numpy.array.
    """

    # if given a TimeSeries instance then get numpy.array
    if isinstance(data, TimeSeries):
        y = data.numpy()
        delta_t = data.delta_t
    else:
        y = data

    # Zero mean
    y = y - y.mean()
    ny_orig = len(y)

    npad = 1
    while npad < 2*ny_orig:
        npad = npad << 1
    ypad = numpy.zeros(npad)
    ypad[:ny_orig] = y

    # FFT data minus the mean
    fdata = TimeSeries(ypad, delta_t=delta_t).to_frequencyseries()

    # correlate
    # do not need to give the congjugate since correlate function does it
    cdata = FrequencySeries(zeros(len(fdata), dtype=fdata.dtype),
                            delta_f=fdata.delta_f, copy=False)
    correlate(fdata, fdata, cdata)

    # IFFT correlated data to get unnormalized autocovariance time series
    acf = cdata.to_timeseries()
    acf = acf[:ny_orig]

    # normalize the autocovariance
    # note that dividing by acf[0] is the same as ( y.var() * len(acf) )
    if unbiased:
        acf /= ( y.var() * numpy.arange(len(acf), 0, -1) )
    else:
        acf /= acf[0]

    # return input datatype
    if isinstance(data, TimeSeries):
        return TimeSeries(acf, delta_t=delta_t)
    else:
        return acf


def calculate_acl(data, m=5, dtype=int):
    r"""Calculates the autocorrelation length (ACL).

    Given a normalized autocorrelation function :math:`\rho[i]` (by normalized,
    we mean that :math:`\rho[0] = 1`), the ACL :math:`\tau` is:

    .. math::

        \tau = 1 + 2 \sum_{i=1}^{K} \rho[i].

    The number of samples used :math:`K` is found by using the first point
    such that:

    .. math::

        m \tau[K] \leq K,

    where :math:`m` is a tuneable parameter (default = 5). If no such point
    exists, then the given data set it too short to estimate the ACL; in this
    case ``inf`` is returned.

    This algorithm for computing the ACL is taken from:

    N. Madras and A.D. Sokal, J. Stat. Phys. 50, 109 (1988).

    Parameters
    -----------
    data : TimeSeries or array
        A TimeSeries of data.
    m : int
        The number of autocorrelation lengths to use for determining the window
        size :math:`K` (see above).
    dtype : int or float
        The datatype of the output. If the dtype was set to int, then the
        ceiling is returned.

    Returns
    -------
    acl : int or float
        The autocorrelation length. If the ACL cannot be estimated, returns
        ``numpy.inf``.
    """

    # sanity check output data type
    if dtype not in [int, float]:
        raise ValueError("The dtype must be either int or float.")

    # if we have only a single point, just return 1
    if len(data) < 2:
        return 1

    # calculate ACF that is normalized by the zero-lag value
    acf = calculate_acf(data)

    cacf = 2 * acf.numpy().cumsum() - 1
    win = m * cacf <= numpy.arange(len(cacf))
    if win.any():
        acl = cacf[numpy.where(win)[0][0]]
        if dtype == int:
            acl = int(numpy.ceil(acl))
    else:
        acl = numpy.inf
    return acl

def batch_acl(samples, nbatches): 
    """
    Performs the batch-mean algorithm to calculate the acl of samples.

    Samples are split up evenly into nbatches batches. For each of those the 
    variance from the mean over all the samples is calculated. Under the 
    assumptions for the Markov chain CLT, the variance of the batch-means from 
    the true mean will be 1/(batch-length)*Var(single sample). 
    Hence the variance of the whole chain is (batch-length)/(nsamples)*s_batches
    """
    mean = samples.mean()
    # Take the highest integer below the fraction
    # This way, will loose at maximum nbatches samples
    batch_length = int(len(samples) // nbatches) 
    batches = [samples[k*batch_length:(k+1)*batch_length] for k in range(nbatches)] 
    batch_means = [] 
    for batch in batches:
        batch_means.append(batch.mean())
    batch_means = numpy.array(batch_means) 
    batch_variance = float(batch_length) / nbatches * numpy.sum((batch_means - mean)**2 )
    print("batch_length is " , batch_length) 
    print("nbatches is " , nbatches)
    print("batch_means are " , batch_means) 
    print("mean is " , mean) 
    print(batch_variance) 
    gamma0 = 1/len(samples) * numpy.sum((samples - samples.mean())**2)
    print("gamma0 is ", gamma0)
    return batch_variance /gamma0

def initial_monotone_sequence(samples): 
    """
    Calculates the normalized initial monotone sequence for the 
    autocovariance function of the samples. 

    The initial monotone sequence is defined as the largest monotone function 
    smaller than Gamma_k = gamma_2k + gamma_(2k+1),
    where gamma_n is the autocovariance with lag n. 
    """
    
    acov = numpy.array(calculate_autocov_function(samples)) 

    n = int(len(samples)/2)
    Gamma = numpy.zeros(n)
    Gamma[:] = acov[:2*n:2] + acov[1:2*n+1:2]
    Gamma_m = numpy.zeros(n)
    Gamma_m[0] = Gamma[0] 
    for i in range(1,n):
        Gamma_m[i] = min(Gamma_m[i-1], Gamma[i])
    assert numpy.all(numpy.diff(Gamma_m)<=0) 
    return Gamma_m / acov[0] 

def calculate_monotone_acl(samples):
    """ 
    For the given 1D array of samples calculates the estimate for the variance 
    from the mean under the assumption of being a stationary markov chain by using
    the initial convex sequence method.
    """ 
    ims = initial_monotone_sequence(samples) 
    maxind = numpy.argmax(ims <= 0) 
    acl = numpy.sum(ims[0:maxind])*2 - 1
    return acl
    