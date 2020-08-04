database=/media/icsl/Samsung_T5/Dataset/KITTI
config=../config
server=username@hostname
password=password
output=/home/icsl/iPython/ChungkeunLee/auto_driving/features_test

for KITTI_dir in $database/*
do
	date=$(echo $KITTI_dir | rev | cut -d"_" -f1 | rev)
	month=$(echo $KITTI_dir | rev | cut -d"_" -f2 | rev)

	for dataset in $KITTI_dir/*
	do
		if [ -d $dataset ]; then
			seq=$(echo $dataset | rev | cut -d"_" -f2 | rev)
			result=2011_${month}_${date}_${seq}

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
done


