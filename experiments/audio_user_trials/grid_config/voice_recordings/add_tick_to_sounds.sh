dirs="alphabet_col_scan_1
      alphabet_col_scan_2 
      alphabet_col_scan_3
      alphabet_col_scan_4
      alphabet_col_scan_5
      alphabet_row_scan"

#Add the tick sounds

for dir in $dirs
	do
		new_dir="${dir}/channels1/"
		a=`ls ${new_dir} --all | grep ogg`
		echo "*************************************************************"
		echo "DIR = ${new_dir}" 
		echo ${a}
		echo "-------------------------------------------------------------"
		for i in $a
			do 
				file_save="${new_dir}${i}"
				sox -m tick_soft.ogg $file_save $file_save
				file_length=`ogginfo $file_save  | grep "Playback length"`
				echo $file_save  $file_length
			done 
		tick_out="${new_dir}tick.ogg"
		cp tick_long.ogg ${tick_out}
		file_length=`ogginfo $tick_out  | grep "Playback length"`
		echo "saving tick_file: ${tick_out}  $file_length"
        done
