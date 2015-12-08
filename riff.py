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
# stupid hack for a bad brain
our_score = np.array(qscores[0][15000:],dtype=np.float32)
if (args.idiot):
    our_score = our_score[:,0:576]
    
mtest = np.array([our_score])
preds = network.predict( mtest )
descs = [quantizer.dl_2_desc(pred) for pred in preds[0]]
# get IDs for anything that's defined
ids = [i for i in range(0,len(our_score)) if ((our_score[i]!=0.0).any())]
defdescs = [descs[i] for i in ids]

input = np.array([our_score[ids]])
# y = network.predict_sequence(mtest[0],100)
for i in range(0,100):
    y = network.predict(input)
    z = quantizer.dl_2_desc(  y[0][-1] )
    print z
    input = np.array([np.concatenate((input[0],y[0][-1:]))])

