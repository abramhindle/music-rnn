import climate
import cPickle as pickle
import theanets
import numpy as np
import logging
import logging.handlers
import argparse
import quantizer

# code clone
parser = argparse.ArgumentParser(description='Estimate MIDI')
parser.add_argument('scores', help='scores',nargs='+')
# parser.add_argument('lin', help='Path of (.midi) label files for training/validation/testing')
#parser.add_argument('tmp', help='Path of directory to write intermediary or data save files',default="./tmp/")
args = parser.parse_args()
scores = args.scores

# code clone
qscores = [quantizer.convert_file(filename) for filename in scores]

network = pickle.load(file("regressor.pkl"))
our_score = np.array(qscores[0][0:1024],dtype=np.float32)
network.predict(np.array([our_score]))
