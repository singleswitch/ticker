#include "audio.h"
#include "fmod_alphabet_player.h"
#include <iostream>

static AlphabetPlayer alphabet_player = AlphabetPlayer();

void playNext(void) { alphabet_player.playNext(); }

int isReady(void) { return alphabet_player.isReady(); }

void setChannels(int i_nchannels) { alphabet_player.setChannels(i_nchannels); }

void restart(void) { alphabet_player.restart(); }

void stop(void) { alphabet_player.stop(); }

void playInstruction(char *i_instruction_name, char *i_file_type)
{
    std::string instruction_name(i_instruction_name);
    std::string file_type(i_file_type);
    alphabet_player.playInstruction(instruction_name, i_file_type);
}

void setRootDir(char *i_dir_name)
{
    std::string dir_name(i_dir_name);
    alphabet_player.setRootDir(dir_name);
}

void setAlphabetDir(char *i_dir_name)
{
    std::string dir_name(i_dir_name);
    alphabet_player.setAlphabetDir(dir_name);
}

void setConfigDir(char *i_dir_name)
{
    std::string dir_name(i_dir_name);
    alphabet_player.setConfigDir(dir_name);
}

void setVolume(float i_val, int i_channel)
{
    alphabet_player.setVolume(i_val, i_channel);
}

int isPlayingInstruction(void)
{
    return (int)alphabet_player.getIsPlayingInstruction();
}

const int *getCurLetterTimes(int *o_size)
{
    const std::vector< int > &letter_times =
        alphabet_player.getCurLetterTimes();
    *o_size = letter_times.size();
    return &letter_times[0];
}

void setTicks(unsigned int i_nticks) { alphabet_player.setTicks(i_nticks); }
