#!/usr/bin/env python2.7

#from common_functions import get_wattlog_from_file, get_timings_from_file, get_time_syscall_pairs_from_file, find_all_syscall_names, list_sorted_subdirectories

#from multiprocessing import Pool

import climate
import re
import cPickle as pickle
import subprocess
import os
import time
import argparse
import datetime
import re
import sys

import math
import theanets
import theano
import numpy as np

import logging
import logging.handlers

#from sklearn import cross_validation
#import matplotlib.pyplot as plt

climate.enable_default_logging()
LOG_FILENAME = 'run_info.log'

logging.basicConfig(filename=LOG_FILENAME,level=logging.INFO)

# Add the log message handler to the logger
handler = logging.handlers.RotatingFileHandler(
    LOG_FILENAME, maxBytes=2**30, backupCount=5)


my_logger = logging.getLogger('GeneralLogger')
my_logger.addHandler(handler)

WINDOW_SIZE = 150

exp = theanets.Experiment(
    theanets.recurrent.Regressor,
    layers=(
        WINDOW_SIZE,
        ('rnn', WINDOW_SIZE),
        ('rnn', WINDOW_SIZE),
        ('rnn', WINDOW_SIZE/2),
        ('rnn', WINDOW_SIZE/4),
        ('rnn', WINDOW_SIZE/8),
        1
    )
)

#exp.load('brains/regressor-2015-11-15T08:02:13.087115') # the brain of the 779th index => start training on 780
vers = 0 # start training on 770 cause we dont have a save

mtrain = [[],[]]
mvalid = [[],[]]



directory = '../calc-strace-time'
bad_counter = 0 + vers # we save weird on the bad_counter - 1 % 10 verions


my_logger.info('begin load file.')
parsed_samples = []
if os.path.isfile('saved_files.mb'):
    parsed_samples = pickle.load(open("saved_files.mb", "rb"))
else:
    print("missing saved_files.p please run the parser.")
    sys.exit(0)
my_logger.info('end load file.')

for sample in parsed_samples: # [train, valid]
    train = sample[0]
    valid = sample[1]
    mtrain[0].append(train[0])
    mtrain[1].append(train[1])
    mvalid[0].append(valid[0])
    mvalid[1].append(valid[1])

BATCH=128
mtrain[0] = np.array(mtrain[0])
mtrain[1] = np.array(mtrain[1])
mvalid[0] = np.array(mvalid[0])
mvalid[1] = np.array(mvalid[1])


# able to call train multiple times -> parse file and train -> GOTO next file. Save every 10 files?
# my_logger.info('begin pretrain')
# exp.train(
#     mtrain,
#     mvalid,
#     algo='layerwise',
#     trainer='nag',
#     patience=100,
#     learning_rate=1e-4,
#     save_progress=("preregressor-{}".format(datetime.datetime.now().isoformat())),
#     save_every=5,
#     #        train_batches=100,
#     batch_size=BATCH,
# )

my_logger.info('begin training')
exp.train(
    mtrain,
    mvalid,
    algo='nag',
    patience=100,
    learning_rate=1e-4,
    max_gradient_norm=10,
    save_progress=("regressor-{}".format(datetime.datetime.now().isoformat())),
    save_every=5,
    #        train_batches=100,
    batch_size=BATCH,
)

exp.save("end-regressor-{}".format(datetime.datetime.now().isoformat()))
