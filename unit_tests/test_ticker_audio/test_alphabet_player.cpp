 
#include "fmod_alphabet_player.h"
#include <iostream>

#ifdef WIN32
#include <windows.h>
#include <conio.h>
#endif
 
//Platform specific libs
#include "../../../fmod/examples/common/wincompat.h" 

int main(int argc, char *argv[])
{ 
    AlphabetPlayer alphabet_player;
    alphabet_player.setAlphabetDir("alphabet_fast/");

    alphabet_player.setRootDir("../../");

    unsigned char key='a';
    alphabet_player.setChannels(5);
	float cur_letter_time = 0;
	int nchannels = 0;
	std::vector<float> letter_start_times; 
	 
    do
    {
        Sleep(20);
        
        if (kbhit())
        {
            key = getch(); 
			if (key == 27) break;
			switch (key) 
			{
				case 'n':
					alphabet_player.playNext();
					break;
				case 'e':
					alphabet_player.stop();
					break;
				case 'r':
					alphabet_player.restart();
					break;	
                // case 't':
				// 	cur_letter_time = alphabet_player.getTime();
				// 	std::cout << "Cur letter time  = " << cur_letter_time << std::endl;
				// 	break;
				case 'k':
					nchannels = alphabet_player.getNChannelsPlaying();
				    std::cout << "Number of channels playing " << nchannels << std::endl;
					break;
				// case 's':
				// 	letter_start_times = alphabet_player.getStartTimes();
				// 	std::cout << "Len letter start times = " << letter_start_times.size() << std::endl;
				// 	for (int n = 0; n < letter_start_times.size(); ++n )
				// 	{
				// 		std::cout << "n = " << n << " time = " << letter_start_times[n] << std::endl;
				// 	}
				// 	break;
			}
			key = 'a';
		}
    } while (key != 27);
	alphabet_player.stop();
    std::cout << "Exit" << std::endl;
    return 0;
}
