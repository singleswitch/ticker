

#ifndef ALPHABET_PLAYER_H
#define ALPHABET_PLAYER_H

#include "fmod.hpp"
#include <fstream>
#include <string>
#include <vector>

// The maximum number of letter groups to play.
#define MAX_ALPHABET 2000
// Maximum number of channels that can be played simultaneously.
#define MAX_CHANNELS 14

class AlphabetPlayer
{
  public:
    AlphabetPlayer(void);

    // Set functions

    // All sounds will stop (call restart to start the sounds)
    void setChannels(int i_nchannels);
    void setRootDir(const std::string i_dir);
    void setAlphabetDir(const std::string i_dir);
    void setConfigDir(const std::string i_dir);
    // Set to channel -1 to change volume of tick sound.
    void setVolume(float i_volume, int i_nchannel);

    // The number of ticks to play before playing alphabet
    void setTicks(unsigned int i_nticks);

    // Get functions

    // Index of the letter that is currently in focus.
    int getCurIndex(void);
    // The number of sounds currently playing.
    int getNChannelsPlaying(void);
    // Is current channel still playing.
    bool getIsCurChannelPlaying(void);
    // File length of current sound playing (in seconds).
    float getCurSoundLength(void);
    // Return if we're playing an instruction.
    bool getIsPlayingInstruction(void);
    // Get the current position of each sound file
    const std::vector< int > &getCurLetterTimes(void);

    // Main functions after setChannels() has been called.

    // Play the alphabet from the beginning of the sequence.
    void restart(void);
    // Stop all the sounds from playing - resume by calling restart.
    void stop(void);
    // Play next sound in cue
    void playNext(void);
    int isReady(void);
    void playInstruction(std::string i_instruction_name,
                         std::string i_file_type);

  private:
    // Init/release sound functions

    // Called by setChannels to reinit everything after the alphabet
    // files have changed.
    void initSoundSystem(void);
    // Free all the memory - called by the destructor and stop.
    void releaseSounds(void);

    // Error checking

    // Error messages are written to std::cerr and a -1 will be
    // returned.
    void fmodErrorCheck(FMOD_RESULT result);

    // File loading functions

    template < class T >
    bool fromString(T &t, const std::string &s,
                    std::ios_base &(*f)(std::ios_base &));

    // Reads characters from current position until newline/space is
    // found.
    template < class T >
    std::string toString(const T i_val);

    template < class T >
    bool readVal(std::ifstream &io_file, T &o_val);

    // Load the stereo position for each channel group.
    void loadGroupStereoPos(const int i_nchannels, std::vector< float > &o_pos);

    // Load the alphabet sequence with "*" indicating spaces (channels
    // to skip).
    void loadAlphabetSequence(const int i_nchannels,
                              std::string &o_alphabet_sequence);
    // Load the letter sound files and their final stereo positions.
    void loadLetterFiles(std::vector< float > &i_channel_pos,
                         std::string &i_alphabet_sequence);

    // Members

    // Configuration variables
    std::string m_root_dir;
    // Directory where the sound recordings can be loaded from.
    std::string m_sound_dir;
    // Directory in voice_recordings directory where alphabet sound
    // files are
    std::string m_alphabet_dir;
    // Directory to find alphabet configuration files in.
    std::string m_config_dir_base;
    // Directory where the sound configurations can be loaded (stereo
    // pos, volume, etc.)
    std::string m_config_dir;
    // Directory containing all the audio instructions.
    std::string m_instructions_dir;
    // All the soundfiles to play (loaded from alphabet sequence).
    std::vector< std::string > m_sound_files;
    // Each sound has a corresponding channel index when playing.
    std::vector< int > m_sound_idx;
    // The current time in each sound file (ms)
    std::vector< int > m_sound_times;
    // The stereo positions of each sound file.
    std::vector< float > m_stereo_pos;
    // The number tick sounds to indicate within-channel rythm.
    unsigned int m_nticks;

    // Indices

    // The index into m_sound_files of current sound.
    int m_cur_sound_index;
    // The index of the channel playing the current sound.
    int m_cur_channel_index;

    // FMOD specific

    // The FMOD status is always stored in this veriable (indicates
    // errors).
    FMOD_RESULT m_result;
    // The FMOD sound streams that are currently playing.
    std::vector< FMOD::Sound * > m_sounds;
    // The FMOD channels allowing manipulation of stereo settings.
    std::vector< FMOD::Channel * > m_channels;
    // Channel to play commands in
    FMOD::Channel *m_instruction_channel;
    // Sound that plays the commands
    FMOD::Sound *m_instruct_sound;
    // The FMOD sound system controlling when to play what.
    FMOD::System *m_p_system;
    // FMOD settings to activate left speaker.
    float m_left_input_on[2];
    // FMOD settings to activate right speaker.
    float m_right_input_on[2];
    // FMOD settings to turn off all speakers.
    float m_input_off[2];
    std::vector< std::vector< float > > m_volume;
};

#endif
