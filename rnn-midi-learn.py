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

parsed_samples = pickle.load(open("saved_files.mb", "rb"))

my_logger.info('begin loading files.')
qscores = [quantizer.convert_file(filename) for filename in scores]
my_logger.info('done loading')

# now take the scores and make training and validation sets
for score in qscores:
    n = len(score)
    s = 3*n/4
    mtrain[0].append(score[0:s-1])
    mtrain[1].append(score[1:s])
    mvalid[0].append(score[s:-1])
    mvalid[1].append(score[s+1:])

BATCH=128

mtrain[0] = np.array(mtrain[0])
mtrain[1] = np.array(mtrain[1])
mvalid[0] = np.array(mvalid[0])
mvalid[1] = np.array(mvalid[1])

hidden_dropout = 0.01
hidden_noise   = 0.01
BATCH=128
activation='sigmoid'
layertype = 'RNN'
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
    patience=10,
    learning_rate=1e-4,
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
    patience=100,
    min_improvement=0.01,
    max_gradient_norm=10,
    learning_rate=1e-4,
    hidden_dropout=hidden_dropout,
    hidden_noise=hidden_noise,
    save_progress=("regressor-{}".format(datetime.datetime.now().isoformat())),
    save_every=5,
    #        train_batches=100,
    batch_size=BATCH,
)

exp.save("end-regressor-{}".format(datetime.datetime.now().isoformat()))
