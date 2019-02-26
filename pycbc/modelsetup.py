"""
Defines some functions to make deeply hidden implementations 
more easily accessible and available to the interactive mode.
"""

import os
import argparse
import logging
import shutil
import h5py

import scipy.optimize
import numpy

import pycbc
from pycbc import (distributions, transforms, fft,
                   opt, psd, scheme, strain, weave)
from pycbc.waveform import generator
from pycbc.types import frequencyseries

from pycbc import __version__
# from pycbc import inference
# from pycbc.inference import (models, burn_in, option_utils)
from pycbc.strain.calibration import Recalibrate

def setup_model_from_arg(arg):
    from pycbc.inference import (models, burn_in, option_utils)
    parser = argparse.ArgumentParser()
    # command line usage
    parser = argparse.ArgumentParser(usage=__file__ + " [--options]",
                                 description=__doc__)
    parser.add_argument("--version", action="version", version=__version__,
                    help="Prints version information.")
    parser.add_argument("--verbose", action="store_true", default=False,
                    help="Print logging messages.")
    # output options
    parser.add_argument("--output-file", type=str, required=True,
                    help="Output file path.")
    parser.add_argument("--force", action="store_true", default=False,
                    help="If the output-file already exists, overwrite it. "
                         "Otherwise, an OSError is raised.")
    parser.add_argument("--save-backup", action="store_true",
                    default=False,
                    help="Don't delete the backup file after the run has "
                         "completed.")
    # parallelization options
    parser.add_argument("--nprocesses", type=int, default=1,
                    help="Number of processes to use. If not given then only "
                         "a single core will be used.")
    parser.add_argument("--use-mpi", action='store_true', default=False,
                    help="Use MPI to parallelize the sampler")
    parser.add_argument("--samples-file", default=None,
                    help="Use an iteration from an InferenceFile as the "
                         "initial proposal distribution. The same "
                         "number of walkers and the same [variable_params] "
                         "section in the configuration file should be used. "
                         "The priors must allow encompass the initial "
                         "positions from the InferenceFile being read.")
    # add data options
    parser.add_argument("--instruments", type=str, nargs="+",
                    help="IFOs, eg. H1 L1.")
    option_utils.add_low_frequency_cutoff_opt(parser)
    parser.add_argument("--psd-start-time", type=float, default=None,
                    help="Start time to use for PSD estimation if different "
                         "from analysis.")
    parser.add_argument("--psd-end-time", type=float, default=None,
                    help="End time to use for PSD estimation if different "
                         "from analysis.")
    parser.add_argument("--seed", type=int, default=0,
                    help="Seed to use for the random number generator that "
                         "initially distributes the walkers. Default is 0.")
    # add config options
    option_utils.add_config_opts_to_parser(parser)
    # add module pre-defined options
    fft.insert_fft_option_group(parser)
    opt.insert_optimization_option_group(parser)
    psd.insert_psd_option_group_multi_ifo(parser)
    scheme.insert_processing_option_group(parser)
    strain.insert_strain_option_group_multi_ifo(parser)
    weave.insert_weave_option_group(parser)
    strain.add_gate_option_group(parser)
    opts = parser.parse_args(arg)
    model_args = dict()

    strain_dict, stilde_dict, psd_dict = option_utils.data_from_cli(opts)
    low_frequency_cutoff_dict = option_utils.low_frequency_cutoff_from_cli(opts)
    if stilde_dict:
        model_args['data'] = stilde_dict 
        # model_args['f_lower'] = low_frequency_cutoff_dict.values()[0]
        # model_args['delta_f'] = stilde_dict.values()[0].delta_f 
        # model_args['delta_t'] = strain_dict.values()[0].delta_t 
        model_args['psds'] = psd_dict 
    
    ctx = scheme.from_cli(opts)


    with ctx: 
        
        cp = option_utils.config_parser_from_cli(opts)
        gates = strain.gates_from_cli(opts)
        if gates:
            model_args['gates'] = gates 

        model = models.read_from_config(cp, **model_args) 
    
    return model

def setup_model(f_min=20,sample_rate=2048,injection_file="injection.hdf",
                config_file="inference.ini",output_file="pseudo_out"):
    TRIGGER_TIME_INT=1126259462
    SEGLEN = 8
    GPS_START_TIME = TRIGGER_TIME_INT - SEGLEN
    GPS_END_TIME = TRIGGER_TIME_INT + SEGLEN 
    arg = ['--seed','12','--asd-file','H1:H1asd.txt','L1:L1asd.txt',
           '--instruments','H1','L1',
           '--gps-start-time',str(GPS_START_TIME),
           '--gps-end-time',str(GPS_END_TIME), 
           '--psd-inverse-length',str(4),
           '--fake-strain','H1:zeroNoise','L1:zeroNoise',
           '--fake-strain-seed',str(44),
           '--strain-high-pass',str(15),
           '--sample-rate',str(sample_rate),
           '--low-frequency-cutoff',str(f_min),
           '--channel-name','H1:FOOBAR','L1:FOOBAR',
           '--injection-file',str(injection_file),
           '--config-file',str(config_file),
           '--output-file',str(output_file),
           '--processing-scheme','cpu',
           '--nprocesses',str(1)]
    return setup_model_from_arg(arg)


def setup_model_from_result(filename):
    f = h5py.File(filename)
    signal = dict()
    psds = dict()
    st = ['approximant', 'f_lower', 'f_ref']
    static_params = {key: f['injections'].attrs[key] for key in st}
    variable_params = list(f.attrs['variable_params'])
    for det in f['data'].keys():
        delta_f = f['data'][det]['stilde'].attrs['delta_f']
        epoch = f['data'][det]['stilde'].attrs['epoch']
        psds[det] = frequencyseries.FrequencySeries(
            f['data'][det]['psds']['0'],
            delta_f, epoch)
        signal[det] = frequencyseries.FrequencySeries(
            f['data'][det]['stilde'],
            delta_f, epoch)
    model = GaussianNoise(variable_params, signal, 20, psds=psds,
                          static_params=static_params)


class model_optimizer:
    """Wraps model to allow optimization by scipy.optimize function.
    """
    def dict_from_array(self, arr):
        """Returns a dict including all parameter values needed for the model,
        given by an ordered input array.
        """
        par = dict()
        for i in range(len(arr)):
            par[self.parameters[i]] = arr[i] 
        return par 
    
    def array_from_dict(self, par):
        """Returns an array with all parameters such that 
        array_from_dict(dict_from_array(arr)) == arr
        """
        arr = numpy.zeros(len(par))
        for i in range(len(arr)):
            arr[i] = par[self.parameters[i]]
        return arr 
    
    def __init__(self, model, par):
        # list of all parameters that need to be specified
        # Must stay constant!! I.e. in the same order!! For dict_from_array!
        self.parameters = tuple(model.variable_params)  
        # start value:
        self.x0 = numpy.array([par[key] for key in self.parameters]) 
        # will be updated: 
        self.x  = self.x0 
        self.model = model
        self.model.sampling_transforms = None
    
    def func(self, arr):
        """Wrapper for model loglikelihood that can be used by the optimizer
        Since the value is minimized, return negative loglikelihood
        """
        par = self.dict_from_array(arr)
        self.model.update(**par)
        return -self.model.loglikelihood

    def optimize(self):
        """Tries to find the minimum of the loglikelihood function
        """ 
        optRes = scipy.optimize.minimize(self.func, self.x, method='Nelder-Mead')
        return optRes

def optimize_model_par(m, par):
    mo = model_optimizer(m, par)
    m.update(**par)
    injection_loglikelihood = m.loglikelihood
    optres = mo.optimize()
    optimal_par = mo.dict_from_array(optres.x)
    m.update(**optimal_par)
    optimal_loglikelihood = m.loglikelihood
    return (injection_loglikelihood, optimal_loglikelihood, 
            par, optimal_par)

def optimize_injection(injection_file, config_file, f_min=20, 
                       sample_rate=2048, output_file="pseudo_out"):
    m = setup_model(f_min=f_min, sample_rate=sample_rate, 
                    injection_file=injection_file, 
                    config_file=config_file, output_file=output_file)
    f = h5py.File(injection_file)
    par = dict(f.attrs.items())
    del par['f_lower']
    del par['f_ref']
    return optimize_model_par(m, par)

def optimize_results(result_file):
    m = setup_model_from_result(result_file)
    f = h5py.File(result_file)
    par = dict(f['injection'].attrs.items())
    del par['f_lower']
    del par['f_ref']
    return optimize_model_par(m, par)

def optimize_to_files(injection_files, config_file, 
                      value_file, parameter_file):
    """
    For a list of injection files generates the models based on 
    all of those files and the config file, calculates the value 
    of the loglikelihood at the injection parameter and finds the 
    optimum close to this value. 
    Then outputs for all injections both the loglikelihood of the 
    injection and the optimal loglikelihood to value_file, and 
    outputs the loglikelihoods and the corresponding parameters 
    to parameter_file.
    """
    injection_loglikelihoods = []
    optimal_loglikelihoods = []
    injection_parameters = []
    optimal_parameters = []
    for injection_file in injection_files:
        il, ol, ip, op = optimize_injection(injection_file, config_file)
        injection_loglikelihoods.append(il)
        optimal_loglikelihoods.append(ol)
        injection_parameters.append(ip)
        optimal_parameters.append(op)
    
    with open(value_file, 'w') as vf:
        for i in range(len(injection_files)):
            vf.write(str(injection_files[i]) + "\n")
            vf.write(str(injection_loglikelihoods[i]) + "\n")
            vf.write(str(optimal_loglikelihoods[i]) + "\n")
            vf.write("\n")
    
    with open(parameter_file, 'w') as pf:
        for i in range(len(injection_files)):
            pad = max([len(t) for t in optimal_parameters[i].keys()] + 
                      [len('loglikelihood')]) + 1
            s = "{0:{pad}} {1:10.3f} {2:10.3f} \n"
            pf.write(injection_files[i] + '\n')
            pf.write(s.format("loglikelihood", 
                              injection_loglikelihoods[i], 
                              optimal_loglikelihoods[i], 
                              pad = pad))
            for key in sorted(optimal_parameters[i].keys()):
                pf.write(s.format(key,
                                  injection_parameters[i][key],
                                  optimal_parameters[i][key], 
                                  pad = pad))
            pf.write('\n')
