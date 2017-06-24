# Slightly dated build file. Refer to README.MD instead.

./clean_up.sh
rm audio.so
python setup.py build
#mv  build/lib.linux-i686-2.6/audio.so ./
mv build/lib.linux-x86_64-2.6/audio.so ./
pyuic4 ticker_layout.ui -o ticker_layout.py
pyuic4 volume_layout.ui -o volume_editor_layout.py
pyuic4 settings_layout.ui -o settings_layout.py
./clean_up.sh
