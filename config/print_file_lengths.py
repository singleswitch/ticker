
import sys
sys.path.append('../')
 
import ogg.vorbis
from channel_config import ChannelConfig  
 
if __name__ ==  "__main__":
    nchannels_list = [5] #1,2,3,4,5] 
    overlap = 0.8  
    file_length_estimate = 0.21
    voice_dir_root  = "../voice_recordings/alphabet_fast/channels"
    for nchannels in nchannels_list:
        print "======================================================="
        print "Computing sound file lengths for nchannels = ", nchannels
        channel_config  = ChannelConfig(nchannels, overlap, file_length_estimate ,  "../", i_display=True ) 
        alphabet = channel_config.getAlphabetLoader().getUniqueAlphabet( i_with_spaces=False)
        voice_dir = voice_dir_root + str(nchannels) + "/"
        for (n, letter) in enumerate(alphabet):
            file_name = voice_dir  + letter + ".ogg"
            file_length = ogg.vorbis.VorbisFile(file_name).time_total(0)*1000.0 
            f_estimate = 1000.0*file_length_estimate 
            print letter, " ", file_length, "ms, estimate = ", f_estimate, "ms", " diff = ", file_length_estimate - f_estimate
