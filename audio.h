#ifndef AUDIO_H
#define AUDIO_H

#ifdef __cplusplus
extern "C" {
#endif
void playNext(void);
int isReady(void);
void setChannels(int i_nchannels);
void restart(void);
void stop(void);
void playInstruction(char *i_instruction_name, char *i_file_type);
void setRootDir(char *i_dir_name);
void setAlphabetDir(char *i_dir_name);
void setConfigDir(char *i_dir_name);
void setVolume(float i_val, int i_channel);
int isPlayingInstruction(void);
const int *getCurLetterTimes(int *o_size);
void setTicks(unsigned int i_nticks);
#ifdef __cplusplus
}
#endif

#endif
