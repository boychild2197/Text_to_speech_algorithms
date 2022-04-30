# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/deepvoice3
'''
import math

def get_Ty(duration, sr, hop_length, r):
    '''Calculates number of paddings for reduction'''
    def _roundup(x):
        return math.ceil(x * .1) * 10
    T = _roundup(duration*sr/hop_length)
    num_paddings = r - (T % r) if T % r != 0 else 0
    T += num_paddings
    return T

class Hyperparams:
    '''Hyper parameters'''
    # signal processing
    sr = 22050 # Sampling rate.
    n_fft = 2048 # fft points (samples)
    frame_shift = 0.0125 # seconds
    frame_length = 0.05 # seconds
    hop_length = int(sr*frame_shift) # samples  This is dependent on the frame_shift.
    win_length = int(sr*frame_length) # samples This is dependent on the frame_length.
    n_mels = 80 # Number of Mel banks to generate
    sharpening_factor = 1.4 # Exponent for amplifying the predicted magnitude
    n_iter = 50 # Number of inversion iterations
    preemphasis = .97 # or None
    max_db = 100
    ref_db = 20

    # Model
    r = 4 # Reduction factor
    dropout_rate = .2
    ## Enocder
    vocab_size = 32 # [PE a-z'.?]
    embed_size = 256 # == e
    enc_layers = 7
    enc_filter_size = 5
    enc_channels = 64 # == c
    ## Decoder
    dec_layers = 4
    dec_filter_size = 5
    attention_size = 128*2 # == a
    ## Converter
    converter_layers = 5*2
    converter_filter_size = 5
    converter_channels = 256 # == v

    sinusoid = False
    attention_win_size = 3

    # data
    data = 'LJSpeech-1.1' # or 'nick (internal)'
    max_duration = 10.0 # seconds
    Tx = 180 # characters. maximum length of text.
    Ty = int(get_Ty(max_duration, sr, hop_length, r)) # Maximum length of sound (frames)

    # training scheme
    lr = 0.001
    logdir = "logdir"
    sampledir = 'samples'
    batch_size = 32
    max_grad_norm = 100.
    max_grad_val = 5.
    num_iterations = 500000

