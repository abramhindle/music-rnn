#!/usr/bin/env python2.7
#
#    Use an RNN to learn some notes
#    Copyright (C) 2015 Stephen Romansky, Abram Hindle
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#

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
import quantizer

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

WINDOW_SIZE = quantizer.TVECSIZE


parser = argparse.ArgumentParser(description='Learn MIDI')
parser.add_argument('scores', help='scores',nargs='+')
# parser.add_argument('lin', help='Path of (.midi) label files for training/validation/testing')
#parser.add_argument('tmp', help='Path of directory to write intermediary or data save files',default="./tmp/")
args = parser.parse_args()
scores = args.scores



mtrain = [[],[]]
mvalid = [[],[]]

# parsed_samples = pickle.load(open("saved_files.mb", "rb"))

my_logger.info('begin loading files.')
qscores = [quantizer.convert_file(filename) for filename in scores]
my_logger.info('done loading')

my_logger.info('Split sets')

# unfortunately we have split stuff up
splitsize = 512

# now take the scores and make training and validation sets
for score in qscores:
    n = len(score)
    for i in range(0,n / splitsize):
        rindex = i * splitsize
        r = rindex + splitsize
        if r+1 > n:
            # if we're over the end just 
            break
        ain  = np.array(score[rindex:r],      dtype=np.float32)
        aout = np.array(score[rindex+1:r+1] , dtype=np.float32)
        if i % 4 == 0:
            # validate
            assignarr = mvalid
        else:
            # train
            assignarr = mtrain
        assignarr[0].append(ain)
        assignarr[1].append(aout)

my_logger.info('Done Split sets: %s, %s' % (len(mtrain[0]),len(mvalid[0])))

    
my_logger.info('Making NP Arrays')
mtrain[0] = np.array(mtrain[0])
mtrain[1] = np.array(mtrain[1])
mvalid[0] = np.array(mvalid[0])
mvalid[1] = np.array(mvalid[1])
my_logger.info('Done')


hidden_dropout = 0.01
hidden_noise   = 0.01
BATCH=64
activation='relu'
layertype = 'LSTM'
exp = theanets.Experiment(
    theanets.recurrent.Regressor,
    layers=(
        WINDOW_SIZE,
        (layertype, WINDOW_SIZE*2,activation),
        (layertype, 3*WINDOW_SIZE/2,activation),
        (layertype, WINDOW_SIZE,activation),
        WINDOW_SIZE
    )
)


# able to call train multiple times -> parse file and train -> GOTO next file. Save every 10 files?
my_logger.info('begin pretrain')
exp.train(
    mtrain,
    mvalid,
    algo='layerwise',
    patience=1,
    learning_rate=1e-3,
    max_gradient_norm=10,
    input_dropout =hidden_dropout,
    hidden_dropout=hidden_dropout,
    #hidden_noise=hidden_noise,
    min_improvement=0.01,
    save_progress=("preregressor-{}".format(datetime.datetime.now().isoformat())),
    save_every=5,
    #        train_batches=100,
    batch_size=BATCH,
)

my_logger.info('begin training')
exp.train(
    mtrain,
    mvalid,
    algo='rmsprop',
    patience=10,
    min_improvement=0.01,
    max_gradient_norm=10,
    learning_rate=1e-3,
    hidden_dropout=hidden_dropout,
    hidden_noise=hidden_noise,
    save_progress=("regressor-{}".format(datetime.datetime.now().isoformat())),
    save_every=5,
    #        train_batches=100,
    batch_size=BATCH,
)

exp.save("end-regressor-{}".format(datetime.datetime.now().isoformat()))
