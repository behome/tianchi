ps -ef | grep class_id | cut -c 9-15| xargs kill -s 9
arr=(8 8 9 15 8 15 19 8 8 9 15 8 15 16 13 7 8)

for i in "${!arr[@]}";
do
    if [ $i -lt 10 ]
    then
      printf "Start class_id %d total %d models on device 0" $i ${arr[$i]}
      nohup python -u code/train.py --model_type=conv --patient=3 --batch_size=256 --checkpoint_path=./user_data/model_data/lstm1/ --class_id=$i --model_num=${arr[$i]} --device=0 > train_"$i".log &
    else
      printf "Start class_id %d total %d models on device 1" $i ${arr[$i]}
      nohup python -u code/train.py --model_type=conv --patient=3 --batch_size=256 --checkpoint_path=./user_data/model_data/lstm1/ --class_id=$i --model_num=${arr[$i]} --device=1 > train_"$i".log &
    fi
    echo
done
