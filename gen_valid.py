import os
import random
import shutil

def move(old_data_path, new_data_path):
	file_name_list = os.listdir(old_data_path)
	tot = int(len(file_name_list) * 0.1)
	while tot > 0:
		n = len(file_name_list)
		i = random.randint(0, n)
		shutil.move(old_data_path + file_name_list[i], new_data_path + file_name_list[i])
		tot -= 1
		file_name_list.remove(file_name_list[i])

move('./train/pos/', './valid/pos/')
move('./train/neg/', './valid/neg/')
