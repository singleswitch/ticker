 
#!/bin/bash

#Notes before running this
ln -s ../../config
ln -s ../../voice_recordings 
make all
cd ../../
python setup.py build
ln -s build/lib*/audio.so

 


 
