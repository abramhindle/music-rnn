# The MIT License (MIT)
# Copyright (c) 2013 Giles F. Hall
# Copyright (c) 2015 Abram Hindle
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# 
import midi
TEMPO = 90
RESOLUTION = 1000
TICK = 60.0/(4*16.0*180)


def note_on(when,channel,note,velocity=60):
    ticktime = int(when/TICK)
    return midi.NoteOnEvent(tick=ticktime, velocity=velocity, pitch=note)

def note_off(when,channel,note,velocity=0):
    ticktime = int(when/TICK)
    return midi.NoteOffEvent(tick=ticktime, velocity=velocity, pitch=note)

def generate_midi(note_tuples,NVOICES=4,TEMPO=TEMPO):
    # Instantiate a MIDI Pattern (contains a list of tracks)
    pattern = midi.Pattern()
    # Instantiate a MIDI Track (contains a list of MIDI events)
    track = midi.Track()
    # Append the track to the pattern
    pattern.append(track)
    # Instantiate a MIDI note on event, append it to the track
    lastwhen = 0
    track.append(midi.SetTempoEvent(bpm=TEMPO))
    for nt in note_tuples:
        t,when,channel,note = nt        
        if t == "on":
            last_note = note_on(when - lastwhen,channel,note)
        elif t == "off":
            last_note = note_off(when - lastwhen,channel,note)
        lastwhen = when
        track.append(last_note)
    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)
    # Print out the pattern
    print pattern
    # Save the pattern to disk
    return pattern

def generate_midi_to_file(filename, note_tuples, NVOICES):
    pattern = generate_midi(note_tuples,NVOICES)
    pattern_to_file(filename, pattern)
    return (filename,pattern)

def pattern_to_file(filename, pattern):
    midi.write_midifile(filename, pattern)

