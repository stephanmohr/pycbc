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