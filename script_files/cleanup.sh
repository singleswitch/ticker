 
#!/bin/bash

message=$1

echo "******************************"
echo "Cleaning all the directories"

rm *~
cd ../
rm talk.txt
rm audio.pyd
rm audio.so
rm -r build
rm *~
rm *.pyc
rm module_tests/test_ticker_audio/config
rm module_tests/test_ticker_audio/voice_recordings


for directory in `find . -type d -maxdepth 5 -mindepth 1 -not -name .hg`
	do
	    
	    echo "cleaning up $directory"  
	    makefiles =`find ${directory}/Makefile`
	    echo "Make file = $makefiles"
	    cd $directory
	    make clean
	    cd -
	    rm -r ${directory}/build
	    rm ${directory}/*.pyc
	    rm ${directory}/*~
	    rm ${directory}/*.so   
	done

echo "******************************"
echo "Hg status after cleanup"
hg status

 


 
