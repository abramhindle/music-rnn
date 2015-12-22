import json
import logging
import numpy as np
import math

TS = 60.0/(4*16.0*180.0) # min timestep
NVOICES = 4 # allow 4 channels
VECSIZE = 128 # size of 1 midi note
TVECSIZE = NVOICES * VECSIZE # size of DL vector
CHANNEL = 3
NOTE = 4
LEN=2
def time2bucket(time):
        ''' quantize time '''
        return int(math.ceil(time/TS))

def line2data(line):
        ''' convert a line (CSV) to desc data: timebuckets and notes (score.pl)'''
        non, noff, channel, note, _ = line.split(',')
        non  = time2bucket(float(non))
        noffb  = time2bucket(float(noff))
        return [non,noffb,float(noff),int(channel),int(note)]

def convert_file(filename):
        ''' convert an entire file to lines then to desc data '''
        logging.info("Loading %s" % filename)
        lines = file(filename).readlines()
        return convert_lines(lines)


def clamp(x,mmin,mmax):
        ''' clamp a value between mmin and mmax '''
        return max(mmin,min(mmax,x))

def first(x):
        return x[0]

def get_max_bucket(descs):
        return max([desc[0]+desc[1] for desc in descs])

def parse_descs(descs):
        max_bucket = get_max_bucket(descs)
        # allow an array of ceil max_time/TS via TVECSIZE
        n = max_bucket + 1
        arr = np.zeros((n,TVECSIZE))
        for desc in descs:
                # [non,noffb,float(noff),int(channel),int(note)]
                non, noffb, noff, channel, note = desc
                column = ((channel-1)%NVOICES)*VECSIZE + note
                start_time = non
                end_time   = noffb + non
                arr[start_time:end_time,column] = 1.0
        return arr

def convert_lines(lines):
        descs = [line2data(str) for str in lines]
        return parse_descs(descs)

def json_eq(a,b):
        ''' equality via json -- slow'''
        return json.dumps(a) == json.dumps(b)        

def note_on(when,channel,instr):
        return ("on",when,channel,instr)

def note_off(when,channel,instr):
        return ("off",when,channel,instr)

def normalize(arr, threshold=0.1):
        arr[arr < threshold] = 0.0
        arr[arr > 0.0] = 1.0
        return arr

def dl_2_events(preds):
        state = TVECSIZE*[False]
        thresh = 0.1
        out = list()
        for i in range(0,len(preds)):
                when = i * TS
                for j in range(0,TVECSIZE):
                        channel = 1 + math.floor(j/VECSIZE)
                        instr   = j % VECSIZE
                        if not state[j] and preds[i][j] > thresh:
                                out.append(note_on(when, channel, instr))
                                state[j] = True
                        elif state[j] and not preds[i][j] > thresh:
                                out.append(note_off(when, channel, instr))
                                state[j] = False
        # close notes
        for j in range(0,TVECSIZE):
                when = len(preds)*TS
                channel = 1 + math.floor(j/VECSIZE)
                instr   = j % VECSIZE
                if state[j]:
                        out.append(note_off(when, channel, instr))

        return out
                                

''' used for test cases '''
convert_lines_test_x = [
        ('%s,%s,1,59,'  % (0,TS)),
        ('%s,%s,2,59,'  % (0,TS)),
        ('%s,%s,1,62,'  % (TS*1,TS*2)),
        ('%s,%s,2,62,'  % (TS*1,TS*2)),
        ('%s,%s,1,67,'  % (TS*1,TS*2)),
        ('%s,%s,2,67,'  % (TS*1,TS*2)),
        ('%s,%s,1,50,'  % (TS*1,TS*3)),
        ('%s,%s,2,50,'  % (TS*1,TS*3)),
        ('%s,%s,1,66,'  % (TS*2,TS*3)),
        ('%s,%s,2,66,'  % (TS*2,TS*4)),
        ('%s,%s,1,43,'  % (TS*3,TS*5)),
        ('%s,%s,2,43,'  % (TS*3,TS*6)),
        ('%s,%s,1,67,'  % (TS*5,TS*6)),
        ('%s,%s,2,67,'  % (TS*5,TS*6)) ]



def convert_lines_test():
        # non, noff, channel, note = str.split(',')
        ts1 = TS
        ts2 = 2*TS
        ts3 = 3*TS
        ts4 = 4*TS
        ts6 = 6*TS
        arr = convert_lines(convert_lines_test_x)
        assert arr[0,59] == 1
        assert arr[0,0] == 0                
        assert arr[0,VECSIZE+59] == 1
        assert arr[1,62] == 1
        assert arr[1,0] == 0                
        assert arr[1,VECSIZE+62] == 1
        assert arr[10,67] == 1
        assert arr[5,0] == 0                
        assert arr[10,VECSIZE+67] == 1
        events = dl_2_events(arr)
        assert events[0][0] == "on"
        assert events[1][0] == "on"
        assert events[0][1] == 0.0
        assert events[1][1] == 0.0
        assert events[0][2] == 1
        assert events[1][2] == 2
        assert events[0][3] == 59
        assert events[1][3] == 59
        print events[-1]
        assert events[-1][0] == "off"
        assert events[-1][3] == 67
        assert events[-1][1] == TS*(6+6) # hmmm
        

def tests():
        assert 1 == clamp(0,1,10)
        assert 10 == clamp(11,1,10)
        assert 5 == clamp(5,1,10)
        assert 15 == clamp(16-1,0,15)
        assert 15 == clamp(16,0,15)
        assert 0 == clamp(-1,0,15)


def run_tests():
        tests()
        convert_lines_test()
        
if __name__ == "__main__":
        run_tests()
        
