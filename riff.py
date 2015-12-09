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
parser.add_argument('-brain',default="regressor.pkl",help="Brain to use")
parser.add_argument('-idiot',default=False,help="Idiot correction")
# parser.add_argument('lin', help='Path of (.midi) label files for training/validation/testing')
#parser.add_argument('tmp', help='Path of directory to write intermediary or data save files',default="./tmp/")
args = parser.parse_args()
scores = args.scores

# code clone
qscores = [quantizer.convert_file(filename) for filename in scores]

network = pickle.load(file(args.brain))
our_score = np.array(qscores[0],dtype=np.float32)
# stupid hack for a bad brain
if (args.idiot):
    our_score = our_score[:,0:576]
    
mtest = np.array([our_score])
preds = network.predict( mtest )
events = quantizer.dl_2_events(preds[0])
print events

import midiit

# from itself
pattern = midiit.generate_midi(events)
midiit.pattern_to_file("pattern.mid",pattern)

def eval_seq( score ):
    mtest = np.array([score])
    preds = network.predict( mtest )
    events = quantizer.dl_2_events(preds[0])
    return (events)
