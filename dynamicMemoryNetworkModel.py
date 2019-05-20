import babi_input_prcoessor
import numpy as np
import tensorflow as tf

from custom_attention_cell import customGRUCellWithAttention

import sys

def positionEncoding(size_of_sentence,size_of_embedding):
	encoding_matrix = np.ones((size_of_embedding,size_of_sentence),dtype = np.float32)
	x = size_of_sentence + 1
	y = size_of_embedding + 1
	for i in range(1,y):
		for j in range(1,x):
			encoding_matrix[i-1,j-1] = (i - (y-1)/2) * (j - (x-1)/2)
		encoding_matrix = 1 + 4*encoding_matrix / size_of_embedding / size_of_sentence
		return np.transpose(encoding_matrix)


class DMN_PLUS(object):
	def data_load(self,debug = False):
		if self.config.train_mode:
			self.train, self.valid, self.word_embedding, self.max_q_len, self.max_sentences, self.max_sen_len, self.vocab_size = babi_input_prcoessor.load_babi(self.config,split_sentences = True)
		else:
			self.test, self.word_embedding, self.max_q_len, self.max_sentences, self.max_sen_len, self.vocab_size = babi_input_prcoessor.load_babi(self.config, split_sentences=True)
		self.encoding = positionEncoding(self.max_sen_len,self.config.embed_size)

	def creating_placeholders(self):
		self.quest_place = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.max_q_len))
		self.inp_place = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.max_sentences, self.max_sen_len))

		self.quest_len_place = tf.placeholder(tf.int32, shape=(self.config.batch_size,))
		self.inp_len_place = tf.placeholder(tf.int32, shape=(self.config.batch_size,))

		self.ans_place = tf.placeholder(tf.int64,shape = (self.config.batch_size,))
		self.dropout_place = tf.placeholder(tf.float32)

	def get_preds(self,out):
		preds = tf.nn.softmax(out)
		pred = tf.argmax(preds,1)
		return pred

	def add_loss(self,out):

		loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=self.ans_place))

		for train_vars in tf.trainable_variables():
			if not 'bias' in train_vars.name.lower():
				loss = loss + self.config.l2*tf.nn.l2_loss(train_vars)
		tf.summary.scalar('loss',loss)

		return loss

	def add_training(self,loss):
		optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr)
		gradients = optimizer.compute_gradients(loss)

		if self.config.cap_grads:
			gradients = [(tf.clip_by_norm(grad, self.config.max_grad_val), var) for grad, var in gradients]
		if self.config.noisy_grads:
			gradients = [(_add_gradient_noise(grad), var) for grad, var in gvs]

		training_operation = optimizer.apply_gradients(gradients)
		return training_operation

	def create_quest_representation(self):

		quest = tf.nn.embedding_lookup(self.embeddings, self.quest_place)
		gatedRUCell = tf.contrib.rnn.GRUCell(self.config.hidden_size)
		_,quest_vector = tf.nn.dynamic_rnn(gatedRUCell,quest,dtype=np.float32,sequence_length=self.quest_len_place)
		return quest_vector

	def create_input_representation(self):
		inps = tf.nn.embedding_lookup(self.embeddings, self.inp_place)

		inps = tf.reduce_sum(inps*self.encoding,2)

		forwardGatedRUCell = tf.contrib.rnn.GRUCell(self.config.hidden_size)
		backwardGatedRUCell = tf.contrib.rnn.GRUCell(self.config.hidden_size)

		outs , _ = tf.nn.bidirectional_dynamic_rnn(forwardGatedRUCell,backwardGatedRUCell,inps,dtype=np.float32,sequence_length=self.inp_len_place)
		factor_vecs = tf.reduce_sum(tf.stack(outs), axis=0)
		factor_vecs = tf.nn.dropout(factor_vecs, self.dropout_place)

		return factor_vecs
	def create_attention(self,quest_vector,prev_mem,factor_vecs,reuse):
		with tf.variable_scope("attention",reuse=reuse):
			features = [factor_vecs*quest_vector,factor_vecs*prev_mem,tf.abs(factor_vecs-quest_vector),tf.abs(factor_vecs-prev_mem)]
			feature_vector = tf.concat(features,1)
			atten = tf.contrib.layers.fully_connected(feature_vector,self.config.embed_size,activation_fn=tf.nn.tanh,reuse=reuse, scope="fc1")
			atten = tf.contrib.layers.fully_connected(atten,1,activation_fn=None,reuse=reuse, scope="fc2")

		return atten

	def generate_episode(self, memory, quest_vector, factor_vecs, hop_index):
		attents = [tf.squeeze(self.create_attention(quest_vector, memory, fv, bool(hop_index) or bool(i)), axis=1)for i, fv in enumerate(tf.unstack(factor_vecs, axis=1))]
		attents = tf.transpose(tf.stack(attents))
		self.attentions.append(attents)
		attents = tf.nn.softmax(attents)
		attents = tf.expand_dims(attents, axis=-1)
		reuse = True if hop_index > 0 else False
		gru_inputs = tf.concat([factor_vecs, attents], 2)
		with tf.variable_scope('attention_gru', reuse=reuse):
			_, episode = tf.nn.dynamic_rnn(customGRUCellWithAttention(self.config.hidden_size),gru_inputs,dtype=np.float32,sequence_length=self.inp_len_place)
		return episode

	def getting_answer_module(self,rnn_output,quest_vec):
		model_output = tf.nn.dropout(rnn_output, self.dropout_place)
		output = tf.layers.dense(tf.concat([model_output, quest_vec], 1),self.vocab_size,activation=None)

		return output

	def inference_module(self):
		with tf.variable_scope("question",initializer=tf.contrib.layers.xavier_initializer()):
			quest_vec = self.create_quest_representation()

		with tf.variable_scope("input",initializer=tf.contrib.layers.xavier_initializer()):
			factor_vecs = self.create_input_representation()

		self.attentions = []

		with tf.variable_scope("memory_module",initializer=tf.contrib.layers.xavier_initializer()):
			prev_memory = quest_vec

			for i in range(self.config.num_hops):

				episode = self.generate_episode(prev_memory,quest_vec,factor_vecs,i)

				with tf.variable_scope("hop_number_"+str(i)):
					prev_memory = tf.layers.dense(tf.concat([prev_memory, episode, quest_vec], 1),self.config.hidden_size,activation=tf.nn.relu)
			output = prev_memory

		with tf.variable_scope("answer_module",initializer=tf.contrib.layers.xavier_initializer()):
			answer_output = self.getting_answer_module(output,quest_vec)
		return answer_output

	def running_training(self, session, data, num_epoch=0, train_writer=None, train_op=None, verbose=2, train=False):
		config = self.config
		dropout = config.dropout

		if train_op is None:
			train_op = tf.no_op()
			dropout = 1

		total_num_steps = len(data[0])
		complete_loss = []
		accuracy = 0

		random_p = np.random.permutation(len(data[0]))
		qp, ip, ql, il, im, a = data
		qp, ip, ql, il, im, a = qp[random_p], ip[random_p], ql[random_p], il[random_p], im[random_p], a[random_p]

		for step in range(total_num_steps):
			index = range(step*config.batch_size,(step+1)*config.batch_size)
			feed = {self.quest_place: qp[index],self.inp_place: ip[index],self.quest_len_place: ql[index],self.inp_len_place: il[index],self.ans_place: a[index],self.dropout_place: dropout}
			loss, pred, summary, _ = session.run([self.calc_loss, self.pred, self.merged, train_op], feed_dict=feed)

			if train_writer is not None:
				train_writer.add_summary(summary, num_epoch*total_num_steps + step)
			
			answers = a[step*config.batch_size:(step+1)*config.batch_size]
			accuracy += np.sum(pred == answers)/float(len(answers))

			complete_loss.append(loss)

			if verbose and step % verbose == 0:
				sys.stdout.write('\r{} / {} : loss = {}'.format(step, total_num_steps, np.mean(complete_loss)))
				sys.stdout.flush()

		if verbose:
			sys.stdout.write('\r')

		output =  np.mean(complete_loss), accuracy/float(total_num_steps)

		return output

	def __init__(self,config):
		self.config = config
		#self.variables_to_save = {}
		self.data_load(debug = False)
		self.creating_placeholders()

		self.embeddings = tf.Variable(self.word_embedding.astype(np.float32), name="Embedding")

		self.output = self.inference_module()
		#print(self.output)
		self.pred = self.get_preds(self.output)
		self.calc_loss = self.add_loss(self.output)
		self.train_step = self.add_training(self.calc_loss)
		self.merged = tf.summary.merge_all()