import climate
import cPickle as pickle
import theanets
import numpy as np
import logging
import logging.handlers
import argparse
import quantizer
import os
import midiit


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



events = quantizer.dl_2_events(our_score)
pattern = midiit.generate_midi(events)
midiit.pattern_to_file("input.mid",pattern)

    
mtest = np.array([our_score])
preds = network.predict( mtest )
events = quantizer.dl_2_events(preds[0])
print events


# from itself
pattern = midiit.generate_midi(events)
midiit.pattern_to_file("pattern.mid",pattern)

def eval_seq( score ):
    mtest = np.array([score])
    preds = network.predict( mtest )
    events = quantizer.dl_2_events(preds[0])
    return (events,preds)

def recursive_eval( score, recurs ):
    mtest = np.array([score])
    for i in range(0,recurs):
        preds = network.predict( mtest )
        mtest = preds
    return mtest

def evolution( score, recurs ):
    mtest = np.array([score])
    for i in range(0,recurs):
        preds = network.predict( mtest )
        mtest = preds
        events = quantizer.dl_2_events(preds[0])
        pattern = midiit.generate_midi(events)
        midiit.pattern_to_file("evolution-{:04d}.mid".format(i),pattern)    
    return mtest

def play_midi_from_arr(preds):
    events = quantizer.dl_2_events(preds[0])
    play_midi(events)

def play_midi(events):
    pattern = midiit.generate_midi(events)
    midiit.pattern_to_file("play.mid",pattern)
    os.system("xterm -e timidity play.mid &")

