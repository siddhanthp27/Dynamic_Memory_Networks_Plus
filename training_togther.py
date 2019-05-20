import os

for i in range(1,21):
	#print(i)
	print('python3 training_dmn.py --babi_task_id '+str(i))
	os.system('python3 training_dmn.py --babiTaskId '+str(i))