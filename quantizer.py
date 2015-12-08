import json
import logging
import numpy as np

TS = 60.0/(4*16.0*180.0) # min timestep
NVOICES = 4 # allow 4 notes 
VECSIZE = 16+127+1 # size of 1 midi note
TVECSIZE = NVOICES * VECSIZE # size of DL vector
CHANNEL = 3
NOTE = 4
LEN=2
def time2bucket(time):
        ''' quantize time '''
        return int(time/TS)

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

def empty_desc():
        return TVECSIZE*[0]

#
#  ts = 1 [fLength,c1,...,c16,n0,...,n127,fLength,c1,...,c16,n0,...,n127,fLength,c1,...,c16,n0,...,n127,fLength,c1,...,c16,n0,...,n127]
#
#
def clamp(x,mmin,mmax):
        ''' clamp a value between mmin and mmax '''
        return max(mmin,min(mmax,x))

def desc_2_dl(descs):
        ''' convert a desc to a list for deep learning '''
        vec = TVECSIZE*[0.0]        
        i = 0
        for desc in descs:
                if i == NVOICES:
                        break
                non,noffb,noff,channel,note = desc
                vec[i*VECSIZE+0] = noff # 1
                vec[i*VECSIZE+1+clamp(channel-1,0,15)] = 1.0 #17
                vec[i*VECSIZE+17+clamp(note,0,127)]    = 1.0
                i += 1
        return vec

def dl_2_desc(dl,threshold=0.01):
        descs = NVOICES * [[0,0,0,0,0]]
        for i in range(0,NVOICES):
                offset = i * VECSIZE
                if (dl[offset] > 0.0):
                        descs[i][CHANNEL] = 1+np.argmax(dl[offset+1:offset+1+16])
                        descs[i][NOTE]    = np.argmax(dl[offset+17:offset+17+128])
                        descs[i][LEN]     = dl[offset] # scale this                        
        return descs


test_vec = None
def dl_2_desc_test():
        global test_vec
        test_vec = np.random.uniform(low=0.0001,high=0.01,size=TVECSIZE)
        for i in range(0,NVOICES):
                test_vec[i*VECSIZE] = 0.0
        vec = test_vec
        descs = dl_2_desc(vec)
        assert len(descs) == NVOICES
        for x in descs:
                print x
                assert x[0] == 0
        # don't test time vec[0] = 1.27866666666667
        vec[0] = 1.0 # it plays
        vec[1] = 0.1 # channel 1
        vec[17+5] = 0.07 # instrument 5
        descs = dl_2_desc(vec)
        assert len(descs) == NVOICES
        print descs[0]
        assert descs[0][CHANNEL] == 1
        assert descs[0][NOTE] == 5
        assert descs[0][LEN] > 0
        vec[VECSIZE] = 0.11 # play it
        vec[VECSIZE+1] = 0.1 # channel 1
        vec[VECSIZE+17+5] = 0.07 # instrument 5
        descs = dl_2_desc(vec)
        assert len(descs) == NVOICES
        assert descs[0][CHANNEL] == 1
        assert descs[0][NOTE] == 5
        assert descs[0][LEN] > 0
        assert descs[1][CHANNEL] == 1
        assert descs[1][NOTE] == 5
        assert descs[1][LEN] > 0
        vec[2*VECSIZE] = 0.11 # play it
        vec[2*VECSIZE+1] = 0.1 # channel 1
        vec[2*VECSIZE+17+6] = 0.07 # instrument 6
        descs = dl_2_desc(vec)
        print descs
        assert len(descs) == NVOICES
        assert descs[0][CHANNEL] == 1
        assert descs[0][NOTE] == 5
        assert descs[0][LEN] > 0
        assert descs[1][CHANNEL] == 1
        assert descs[1][NOTE] == 5
        assert descs[1][LEN] > 0
        assert descs[2][CHANNEL] == 1
        assert descs[2][NOTE] == 6
        assert descs[2][LEN] > 0
        vec[3*VECSIZE] = 0.11 # play it
        vec[3*VECSIZE+1] = 0.1 # channel 1
        vec[3*VECSIZE+17+9] = 0.07 # instrument 9
        descs = dl_2_desc(vec)
        assert len(descs) == NVOICES
        assert descs[0][CHANNEL] == 1
        assert descs[0][NOTE] == 5
        assert descs[0][LEN] > 0
        assert descs[1][CHANNEL] == 1
        assert descs[1][NOTE] == 5
        assert descs[1][LEN] > 0
        assert descs[2][CHANNEL] == 1
        assert descs[2][NOTE] == 6
        assert descs[2][LEN] > 0
        assert descs[3][CHANNEL] == 1
        assert descs[3][NOTE] == 9
        assert descs[3][LEN] > 0

        
        
                


                
def first(x):
        return x[0]

def group_lines(descs):
        ''' group notes at the same time together '''
        descs = sorted(descs, key=first)
        out = list()
        curr = list()
        for desc in descs:
                # this could be simplified
                if len(curr) > 0:
                        if desc[0] == curr[0][0]:
                                curr.append(desc)
                        else:
                                out.append(curr)
                                curr = list()
                                curr.append(desc)
                else:
                        curr.append(desc)
        if len(curr) > 0:
                out.append(curr)
        return out
                
def parse_descs(descs):
        hdescs = insert_empty_groups(group_lines(descs))
        vectors = [desc_2_dl(descs) for descs in hdescs]
        return vectors

def convert_lines(lines):
        descs = [line2data(str) for str in lines]
        return parse_descs(descs)

def json_eq(a,b):
        ''' equality via json -- slow'''
        return json.dumps(a) == json.dumps(b)        

def insert_empty_groups(groups):
        ''' damn time series and 0 measures -- this method adds empties to missing timesteps'''
        time = 0
        out = list()
        i = 0
        while(i < len(groups)):
                if len(groups[i]) > 0 and time == groups[i][0][0]:
                        out.append(groups[i])
                        i+=1                        
                else:
                        out.append(list())
                time += 1
        return out

''' used for test cases '''
convert_lines_test_x = [
        ('%s,1.27866666666667,1,59,'  % (0)),
        ('%s,1.27866666666667,2,59,'  % (0)),
        ('%s,0.158666666666667,1,62,' % (TS*1+0.0001)),
        ('%s,0.158666666666667,2,62,' % (TS*1+0.0001)),
        ('%s,0.158666666666667,1,67,' % (TS*1+0.0001)),
        ('%s,0.158666666666667,2,67,' % (TS*1+0.0001)),
        ('%s,0.318666666666667,1,50,' % (TS*1+0.0001)),
        ('%s,0.318666666666667,2,50,' % (TS*1+0.0001)),
        ('%s,0.158666666666667,1,66,' % (TS*2+0.0001)),
        ('%s,0.158666666666667,2,66,' % (TS*2+0.0001)),
        ('%s,0.318666666666667,1,66,' % (TS*2+0.0001)),
        ('%s,0.318666666666667,2,66,' % (TS*2+0.0001)),
        ('%s,0.638666666666667,1,43,' % (TS*3+0.0001)),
        ('%s,0.638666666666667,2,43,' % (TS*3+0.0001)),
        ('%s,0.318666666666667,1,67,' % (TS*5+0.0001)),
        ('%s,0.318666666666667,2,67,' % (TS*5+0.0001)) ]



                

def convert_lines_test():
        # non, noff, channel, note = str.split(',')
        ts1 = TS
        ts2 = 2*TS
        ts3 = 3*TS
        ts4 = 4*TS
        ts6 = 6*TS
        vectors = convert_lines(convert_lines_test_x)
        assert vectors[0][0] == 1.27866666666667
        assert vectors[0][VECSIZE] == 1.27866666666667
        assert vectors[0][17+59] == 1.0
        assert vectors[1][17+62] == 1.0
        assert vectors[2][17+66] == 1.0
        assert [len(x) for x in vectors] == 6*[TVECSIZE]
        assert vectors[3][17+43] == 1.0
        assert vectors[4][17+43] == 0.0 # test blank row
        assert vectors[4][17+67] == 0.0 # test blank row
        assert vectors[5][17+67] == 1.0
        

def tests():
        assert 1 == clamp(0,1,10)
        assert 10 == clamp(11,1,10)
        assert 5 == clamp(5,1,10)
        assert 15 == clamp(16-1,0,15)
        assert 15 == clamp(16,0,15)
        assert 0 == clamp(-1,0,15)
        data  = [[1],[1],[1],[2]]
        odata = [[[1],[1],[1]],[[2]]]
        odata_p = group_lines(data)
        assert json_eq(odata_p,odata)
        data2  = [[1],[2],[3],[4]]
        odata2  = [[[1]],[[2]],[[3]],[[4]]]
        odata_p = group_lines(data2)
        assert json_eq(odata_p,odata2)
        oarr = TVECSIZE*[0.0]
        oarr[0*VECSIZE] = 1.0
        oarr[0*VECSIZE+16] = 1.0 # channel 16
        oarr[0*VECSIZE+17] = 1.0 # note 0
        desc1 = [0,int(1.0/TS),1.0,16,0]
        desc2 = [0,int(1.0/TS),1.0,2,31]
        descs = [desc1]
        assert json_eq(oarr,desc_2_dl(descs))
        descs = [desc1,desc2]
        oarr[1*VECSIZE] = 1.0
        oarr[1*VECSIZE+1+2-1] = 1.0 # channel 2
        oarr[1*VECSIZE+17+31] = 1.0 # note 31
        assert json_eq(oarr,desc_2_dl(descs))
        assert line2data('0.00520833333333,1.27866666666667,1,59,') == [0, 245, 1.27866666666667, 1, 59]
        descs = [desc1,desc2,desc2,desc2]
        oarr[2*VECSIZE] = 1.0
        oarr[2*VECSIZE+1+2-1] = 1.0 # channel 2
        oarr[2*VECSIZE+17+31] = 1.0 # note 31
        oarr[3*VECSIZE] = 1.0
        oarr[3*VECSIZE+1+2-1] = 1.0 # channel 2
        oarr[3*VECSIZE+17+31] = 1.0 # note 31
        # test 4 voices
        assert json_eq(oarr,desc_2_dl(descs))
        # test more voices!
        descs = [desc1,desc2,desc2,desc2,desc2,desc2]
        assert json_eq(oarr,desc_2_dl(descs))


def run_tests():
        tests()
        convert_lines_test()
        dl_2_desc_test()
        
if __name__ == "__main__":
        run_tests()
        
