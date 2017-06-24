dirs="alphabet_fast"

#Add the tick sounds

for dir in $dirs
	do
		new_dir="${dir}/channels4/"
		a=`ls ${new_dir} --all | grep ogg`
		echo "*************************************************************"
		echo "DIR = ${new_dir}" 
		echo ${a}
		echo "-------------------------------------------------------------"
		for i in $a
			do 
				file_save="${new_dir}${i}"
				#sox $file_save $file_save tempo 1.25 #1.9
				file_length=`ogginfo $file_save  | grep "Playback length"`
				echo $file_save  $file_length
			done 
        done
