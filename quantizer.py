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
        
