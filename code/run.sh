ps -ef | grep class_id | cut -c 9-15| xargs kill -s 9
python code/test.py
