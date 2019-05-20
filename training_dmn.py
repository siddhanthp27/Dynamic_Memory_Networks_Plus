import time
import tensorflow as tf
import os
import argparse
from Config import Config

from dynamicMemoryNetworkModel import DMN_PLUS

class argument_parser():
	def __init__(self,parser):
		self.parser = parser
		self.parser.add_argument("-b","--babiTaskId")
		self.parser.add_argument("-r","--restore")
		self.parser.add_argument("-s","--strongSupervision")
		self.parser.add_argument("-t","--dynamicmemoryType")
		self.parser.add_argument("-l","--l2_loss",type = float,default = 0.001)
		self.parser.add_argument("-n","--num_runs",type = int)
	def returnFinalParser(self):
		return self.parser.parse_args()

parser = argparse.ArgumentParser()
argObject = argument_parser(parser)
args = argObject.returnFinalParser()

dynamic_memory_type = args.dynamicmemoryType if args.dynamicmemoryType is not None else "plus"

if dynamic_memory_type == "plus":
	config = Config()
	print('Dynamic Memory Type : '+dynamic_memory_type)
else:
	print(dynamic_memory_type+'Is not implemented yet')
	exit()

if args.babiTaskId is not None:
	config.babi_id = args.babiTaskId

num_runs = args.num_runs if args.num_runs is not None else 1

print('Training Dynamic Memory Network ' + dynamic_memory_type + ' on babi-task ID : ', config.babi_id)

with tf.variable_scope('Dynamic_Memory_Network') as scope:
	dmn_model = DMN_PLUS(config)

for run in range(num_runs):
	print('Starting Run number : ',run)

	init = tf.global_variables_initializer()
	saver = tf.train.Saver()

	with tf.Session() as sess:
		summary_directory = 'summaries/train/' + time.strftime("%Y-%m-%d %H %M")
		if not os.path.exists(summary_directory):
			os.makedirs(summary_directory)
		train_values_writer = tf.summary.FileWriter(summary_directory,sess.graph)

		sess.run(init)
		best_val_epoch = 0
		prev_epoch_loss = float('inf')
		best_val_loss = float('inf')
		best_val_accuracy = 0.0
		best_overall_val_loss = float('inf')
		if args.restore:
			saver.restore(sess, 'weights/task' + str(dmn_model.config.babi_id) + '.weights')
		for epoch in range(config.max_epochs):
			starting_time = time.time()
			train_loss, train_accuracy = dmn_model.running_training(sess, dmn_model.train, epoch, train_values_writer,train_op=dmn_model.train_step, train=True)
			valid_loss, valid_accuracy = dmn_model.running_training(sess, dmn_model.valid)
			print('Training Loss : '+str(train_loss))
			print('Validation Loss : '+str(valid_loss))
			print('Training Accuracy : '+str(train_accuracy))
			print('Validation Accuracy : '+str(valid_accuracy))

			if valid_loss < best_val_loss:
				best_val_loss = valid_loss
				best_val_epoch = epoch
				if best_val_loss < best_overall_val_loss:
					best_overall_val_loss = best_val_loss
					best_val_accuracy = valid_accuracy
					saver.save(sess,'weights/task'+str(dmn_model.config.babi_id) + '.weights')
			if train_loss > prev_epoch_loss*dmn_model.config.anneal_threshold:
				model.config.lr = model.config.lr / model.config.anneal_by
			prev_epoch_loss = train_loss

			if epoch - best_val_epoch > config.early_stopping:
				print('Total time',time.time()-start)
				break
			print('Total time',time.time()-start)
		print('Validation Accuracy: ',best_val_accuracy)