
rm -r ./build/*

dirs="./ experiments/user_trials/ experiments/simulations/ experiments/simulations/deprecated/ voice_recordings/record_alphabet/ experiments/audio_user_trials/ config/ experiments/multi_channel_user_trials/ experiments/ alphabet_sequences/"

for d in $dirs
  do
     rm ${d}*.pyc
     rm ${d}*~
     rm ${d}*.py.orig
  done
echo "*********************************"
echo "Mercurial results"
hg status

