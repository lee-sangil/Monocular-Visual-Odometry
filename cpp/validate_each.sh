database=/media/icsl/Samsung_T5/Dataset/KITTI
config=../config
server=username@hostname
password=password
output=/home/icsl/iPython/ChungkeunLee/auto_driving/features_depth

array_seq=(	"2011_09_26_drive_0056_sync"
			"2011_09_26_drive_0059_sync")

for dataset in ${array_seq[@]}
do
	seq=$(echo $dataset | rev | cut -d"_" -f2 | rev)
	date=$(echo $dataset | rev | cut -d"_" -f4 | rev)
	month=$(echo $dataset | rev | cut -d"_" -f5 | rev)
	result=2011_${month}_${date}_${seq}

	dataset=$database/2011_${month}_${date}/$dataset

	if [ -d $dataset ]; then
		./mono -i $dataset/ -c $config/kitti_$month$date.yaml -script

		for idx in {1..10}
		do
			sshpass -p $password ssh $server '[ -d '$output/${result}_$idx' ]'
			if [[ $? -eq 1 ]]; then
				ssh_output=$output/${result}_$idx
				sshpass -p $password ssh $server mkdir $ssh_output
				break
			fi
		done

		sshpass -p $password scp features/* $server:$ssh_output/
	fi
done


