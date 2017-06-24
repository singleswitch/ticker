#include "fmod_alphabet_player.h"
#include "fmod_errors.h"
#include <iostream>
#include <sstream>

//########################################### Initialisation/Release functions
AlphabetPlayer::AlphabetPlayer()
{
    m_sound_files.reserve(MAX_ALPHABET);
    m_stereo_pos.reserve(MAX_ALPHABET);
    m_sound_times.reserve(MAX_ALPHABET);
    m_p_system = NULL;
    m_channels.resize(MAX_CHANNELS);
    m_sound_idx.resize(MAX_CHANNELS);
    m_left_input_on[0] = 1;
    m_left_input_on[1] = 0;
    m_right_input_on[0] = 0;
    m_right_input_on[1] = 1;
    m_input_off[0] = 0;
    m_input_off[1] = 0;
    m_cur_sound_index = -1;
    m_cur_channel_index = -1;
    m_instruct_sound = NULL;
    m_config_dir_base = "config/channels";
    m_alphabet_dir = "alphabet/";
    setRootDir("./");
    m_instruction_channel = NULL;
    m_nticks = 2;
}

void AlphabetPlayer::initSoundSystem(void)
{
    // NB: This function assumes all the pointers (ie. to system and
    // sounds have been released).

    // Create the sound system
    releaseSounds();
    m_result = FMOD::System_Create(&m_p_system);
    fmodErrorCheck(m_result);
    m_result = m_p_system->setOutput(FMOD_OUTPUTTYPE_ALSA);
    fmodErrorCheck(m_result);
    m_result = m_p_system->init(32, FMOD_INIT_NORMAL, 0);
    fmodErrorCheck(m_result);
    // Init all the sounds
    m_sounds.resize(m_sound_files.size());
    m_sound_times.resize(m_sound_files.size());
    for (unsigned int n = 0; n < m_sound_files.size(); ++n)
    {
        m_sounds[n] = NULL;
        m_sound_times[n] = -1;
    }
    // Init all indices
    m_cur_sound_index = -1;
    m_cur_channel_index = -1;
    // Load the sounds
    for (unsigned int n = 0; n < m_sound_files.size(); ++n)
    {
        m_result = m_p_system->createStream(m_sound_files[n].c_str(),
                                            FMOD_SOFTWARE | FMOD_LOOP_OFF, 0,
                                            &m_sounds[n]);
        fmodErrorCheck(m_result);
    }
}

void AlphabetPlayer::stop(void) { releaseSounds(); }

void AlphabetPlayer::restart(void) { initSoundSystem(); }

void AlphabetPlayer::releaseSounds(void)
{
    if (m_instruct_sound != NULL)
    {
        m_result = m_instruct_sound->release();
        fmodErrorCheck(m_result);
        m_instruct_sound = NULL;
    }

    m_instruction_channel = NULL;

    for (unsigned int n = 0; n < m_sounds.size(); ++n)
    {
        if (m_sounds[n] != NULL)
        {
            m_result = m_sounds[n]->release();
            fmodErrorCheck(m_result);
            m_sounds[n] = NULL;
            m_sound_times[n] = -1;
        }
    }

    for (int n = 0; n < MAX_CHANNELS; ++n)
    {
        if (m_channels[n] != NULL)
        {
            m_channels[n]->stop();
            m_channels[n] = NULL;
            m_sound_idx[n] = -1;
        }
    }

    if (m_p_system != NULL)
    {
        m_result = m_p_system->close();
        fmodErrorCheck(m_result);
        m_result = m_p_system->release();
        fmodErrorCheck(m_result);
        m_p_system = NULL;
    }
}
//######################################################### Error Checks
void AlphabetPlayer::fmodErrorCheck(FMOD_RESULT result)
{
    if (result != FMOD_OK)

    {
        std::cerr << "FMOD error! (" << result << ") "
                  << FMOD_ErrorString(result);
        // exit(-1);
    }
}
//############################################# File loading functions

template <class T>
bool AlphabetPlayer::fromString(T &t, const std::string &s,
                                std::ios_base &(*f)(std::ios_base &))
{
    /*Examples:1.  if(from_string<int>(i, std::string("ff"), std::hex))  ...
             2.  if(from_string<float>(f, std::string("123.456"), std::dec)) ...
     */
    std::istringstream iss(s);
    return !(iss >> f >> t).fail();
}

template <class T>
std::string AlphabetPlayer::toString(const T i_val)
{
    std::stringstream out;
    out << i_val;
    return out.str();
}

template <class T>
bool AlphabetPlayer::readVal(std::ifstream &i_file, T &o_val)
{
    std::string cur_string = "";
    char cur_letter = '0';
    bool assigned = false;
    while (i_file.good())
    {
        cur_letter = (char)i_file.get();
        if ((cur_letter == ' ') || (cur_letter == '\n'))
        {
            if (!assigned)
            {
                return false;
            }
            return fromString<T>(o_val, cur_string, std::dec);
        }
        cur_string += cur_letter;
        assigned = true;
    }
    return assigned;
}

void AlphabetPlayer::loadAlphabetSequence(const int i_nchannels,
                                          std::string &o_alphabet_sequence)
{
    std::string nchannels = toString<int>(i_nchannels);
    std::string config_file = m_config_dir + nchannels + "/alphabet.txt";
    std::ifstream infile;
    infile.open(config_file.c_str(), std::ios_base::in);
    char cur_letter;
    o_alphabet_sequence = "";
    while (infile.good())
    {
        cur_letter = (char)infile.get();
        if ((cur_letter == ' ') || (cur_letter == '\n'))
        {
            break;
        }
        o_alphabet_sequence += cur_letter;
    }
    infile.close();
}

void AlphabetPlayer::loadGroupStereoPos(const int i_nchannels,
                                        std::vector<float> &o_pos)
{
    std::string nchannels = toString<int>(i_nchannels);
    std::string config_file = m_config_dir + nchannels + "/stereo_pos.txt";
    std::ifstream infile;
    infile.open(config_file.c_str(), std::ios_base::in);
    o_pos.resize(i_nchannels);
    for (int n = 0; n < i_nchannels; ++n)
    {
        if (!readVal<float>(infile, o_pos[n]))
        {
            std::cerr << "Can not read stereo position from " << config_file
                      << " channel = " << n << std::endl;
            infile.close();
            // exit(-1);
            return;
        }
    }
    infile.close();
}

void AlphabetPlayer::loadLetterFiles(std::vector<float> &i_channel_pos,
                                     std::string &i_alphabet_sequence)
{
    unsigned int nchannels = i_channel_pos.size();
    std::string sound_dir = m_sound_dir + toString<int>(nchannels) + "/";
    m_sound_files.clear();
    m_sound_times.clear();
    m_stereo_pos.clear();
    m_volume.resize(nchannels);

    for (unsigned int n = 0; n < m_volume.size(); ++n)
    {
        m_volume[n].clear();
    }
    /* - Add the tick noise to the beginning of the first channel
     - The actual stero location will be contained in m_stero_pos*/
    for (unsigned int n = 0; n < m_nticks; ++n)
    {
        m_sound_files.push_back(sound_dir + "tick.ogg");
        m_stereo_pos.push_back(0.0);
        m_volume[0].push_back(1.0);
    }
    /* Add -1 as placeholders to channels 2-nchannels*/
    for (unsigned int n = 1; n < nchannels; ++n)
    {
        for (unsigned int m = 0; m < m_nticks; ++m)
        {
            m_volume[n].push_back(-1.0);
        }
    }
    // The rest of the letters
    for (unsigned int n = 0, channel_cnt = 0; n < i_alphabet_sequence.length();
         ++n, ++channel_cnt)
    {
        while (channel_cnt >= nchannels)
            channel_cnt -= nchannels;

        if (i_alphabet_sequence[n] == '*')
        {
            m_volume[channel_cnt].push_back(-1.0);
            continue;
        }
        m_sound_files.push_back(sound_dir + i_alphabet_sequence[n] + ".ogg");
        m_stereo_pos.push_back(i_channel_pos[channel_cnt]);
        m_volume[channel_cnt].push_back(1.0);
    }

    /*for (unsigned int c = 0; c < m_volume.size(); c++)
  {
    for (unsigned int l = 0; l < m_volume[c].size(); l++)
    {
      if (m_volume[c][l] < 0)
      {
                std::cout << 0;
      }
      else
      {
                std::cout << m_volume[c][l];
      }
    }
    std::cout << std::endl;
  }*/
}
//######################################### The main functions

void AlphabetPlayer::setTicks(unsigned int i_nticks) { m_nticks = i_nticks; }

void AlphabetPlayer::setChannels(int i_nchannels)
{
    releaseSounds();
    // Load the phase offset of channel (generates the stereo effect).
    std::vector<float> group_stereo_pos;
    loadGroupStereoPos(i_nchannels, group_stereo_pos);
    // Load the sequence in which letters should be read.
    std::string alphabet_sequence = "";
    loadAlphabetSequence(i_nchannels, alphabet_sequence);
    // Load the letter sound files and their final stereo positions.
    loadLetterFiles(group_stereo_pos, alphabet_sequence);
    // Setup the whole sound system.
    initSoundSystem();
    m_p_system->update();
}

void AlphabetPlayer::setRootDir(const std::string i_dir)
{
    m_root_dir = i_dir;
    m_sound_dir = i_dir + "voice_recordings/" + m_alphabet_dir + "channels";
    m_config_dir = i_dir + m_config_dir_base;
    m_instructions_dir = i_dir + "voice_recordings/commands/";
}

void AlphabetPlayer::setConfigDir(const std::string i_dir)
{
    m_config_dir_base = i_dir;
    setRootDir(m_root_dir);
}

void AlphabetPlayer::setAlphabetDir(const std::string i_dir)
{
    m_alphabet_dir = i_dir;
    setRootDir(m_root_dir);
}

int AlphabetPlayer::isReady(void)
{
    bool is_playing = false;
    for (unsigned int n = 0; n < m_channels.size(); ++n)
    {
        if (m_channels[n] == NULL)
            continue;
        m_channels[n]->isPlaying(&is_playing);
        if (is_playing)
            return 0;
    }
    if (m_instruction_channel == NULL)
    {
        return 1;
    }
    m_instruction_channel->isPlaying(&is_playing);
    if (is_playing)
        return 0;
    return 1;
}

void AlphabetPlayer::playNext(void)
{
    ++m_cur_sound_index;
    if (((unsigned int)m_cur_sound_index) >= m_sounds.size())
        if (((unsigned int)m_cur_sound_index) >= m_sounds.size())
        {
            m_cur_sound_index = 0;
        }
    ++m_cur_channel_index;
    if (((unsigned int)m_cur_channel_index) >= m_channels.size())
    {
        m_cur_channel_index = 0;
    }
    m_sound_idx[m_cur_channel_index] = m_cur_sound_index;
    m_result =
        m_p_system->playSound(FMOD_CHANNEL_FREE, m_sounds[m_cur_sound_index],
                              false, &m_channels[m_cur_channel_index]);
    fmodErrorCheck(m_result);

    // Get the current channel and letter index
    int channel = -1;
    int letter_index = 0;
    int cnt_sound = 0;
    while (cnt_sound <= m_cur_sound_index)
    {
        if (++channel >= ((int)m_volume.size()))
        {
            channel = 0;
            ++letter_index;
        }
        if (m_volume[channel][letter_index] > 0.0)
            ++cnt_sound;
    }

    // std::cout << "sound index = " << m_cur_sound_index << " channel = " <<
    // channel << " letter index = " << letter_index <<  " volume = "
    //    << m_volume[channel][letter_index] << " pan = " <<
    //    m_stereo_pos[m_cur_sound_index]  <<  std::endl;

    m_result = m_channels[m_cur_channel_index]->setVolume(
        m_volume[channel][letter_index]);
    fmodErrorCheck(m_result);
    // FIXME: THIS LOOKS WEIRD BUT SOUNDS RIGHT, CHECK
    m_result = m_channels[m_cur_channel_index]->setSpeakerLevels(
        FMOD_SPEAKER_FRONT_LEFT, m_left_input_on, 3);
    fmodErrorCheck(m_result);
    m_result = m_channels[m_cur_channel_index]->setSpeakerLevels(
        FMOD_SPEAKER_FRONT_LEFT, m_right_input_on, 3);
    fmodErrorCheck(m_result);
    m_result = m_channels[m_cur_channel_index]->setSpeakerLevels(
        FMOD_SPEAKER_FRONT_LEFT, m_input_off, 3);
    fmodErrorCheck(m_result);
    m_result = m_channels[m_cur_channel_index]->setPan(
        m_stereo_pos[m_cur_sound_index]);
    m_p_system->update();
}

void AlphabetPlayer::setVolume(float i_volume, int i_nchannel)
{
    if (i_nchannel >= ((int)m_channels.size()))
    {
        std::cerr << "Can not set volume of channel " << i_nchannel << ", only "
                  << m_channels.size() - 1 << " allowed " << std::endl;
        return;
    }
    unsigned int start_index = 0, nchannel = 0;
    if (i_nchannel >= 0)
    {
        start_index = m_nticks;
        nchannel = (unsigned int)i_nchannel;
    }
    for (unsigned int n = start_index; n < m_volume[nchannel].size(); ++n)
    {
        if (m_volume[nchannel][n] < 0.0)
            continue;
        m_volume[nchannel][n] = i_volume;
    }
}

float AlphabetPlayer::getCurSoundLength(void)
{
    if (m_cur_sound_index < 0)
        return 0.0;
    unsigned int sound_end;
    m_result =
        m_sounds[m_cur_sound_index]->getLength(&sound_end, FMOD_TIMEUNIT_MS);
    return 0.001f * ((float)sound_end);
}

int AlphabetPlayer::getNChannelsPlaying(void)
{
    if (m_cur_sound_index < 0)
        return 0;
    int n_channels_playing = 0;
    m_result = m_p_system->getChannelsPlaying(&n_channels_playing);
    fmodErrorCheck(m_result);
    return n_channels_playing;
}

int AlphabetPlayer::getCurIndex(void) { return m_cur_sound_index; }

bool AlphabetPlayer::getIsCurChannelPlaying(void)
{
    if (m_cur_sound_index < 0)
        return false;
    bool is_playing = false;
    m_result = m_channels[m_cur_channel_index]->isPlaying(&is_playing);
    return is_playing;
}

const std::vector<int> &AlphabetPlayer::getCurLetterTimes(void)
{
    unsigned int letter_time = 0;
    bool is_playing = false;
    m_sound_times.assign(m_sound_times.size(), -1);

    // If a sound is playing set it's corresponding letter time
    for (unsigned int n = 0; n < m_channels.size(); ++n)
    {
        if (m_channels[n] == NULL)
            continue;
        m_channels[n]->isPlaying(&is_playing);
        if (is_playing)
        {
            m_result =
                m_channels[n]->getPosition(&letter_time, FMOD_TIMEUNIT_MS);
            fmodErrorCheck(m_result);
            m_sound_times[m_sound_idx[n]] = letter_time;
        }
    }
    return m_sound_times;
}

bool AlphabetPlayer::getIsPlayingInstruction(void)
{
    if (m_instruction_channel == NULL)
    {
        return false;
    }
    bool is_playing;
    m_instruction_channel->isPlaying(&is_playing);
    return (int)is_playing;
}

void AlphabetPlayer::playInstruction(std::string i_instruction_name,
                                     std::string i_file_type)
{

    std::string sound_file(m_instructions_dir + i_instruction_name +
                           i_file_type);
    m_result = m_p_system->createSound(sound_file.c_str(),
                                       FMOD_SOFTWARE | FMOD_LOOP_OFF, 0,
                                       &m_instruct_sound);
    fmodErrorCheck(m_result);
    m_result = m_p_system->playSound(FMOD_CHANNEL_FREE, m_instruct_sound, false,
                                     &m_instruction_channel);
    fmodErrorCheck(m_result);
}
