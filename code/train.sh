ps -ef | grep class_id | cut -c 9-15| xargs kill -s 9
arr=(4 4 5 8 4 8 9 4 4 5 8 4 8 8 6 3 4)

for i in "${!arr[@]}";
do
    if [ $i -lt 10 ]
    then
      printf "Start class_id %d total %d models on device 0" $i ${arr[$i]}
      nohup python -u code/train.py --class_id=$i --model_num=${arr[$i]} --device=0 > train_"$i".log &
    else
      printf "Start class_id %d total %d models on device 1" $i ${arr[$i]}
      nohup python -u code/train.py --class_id=$i --model_num=${arr[$i]} --device=1 > train_"$i".log &
    fi
    echo
done
