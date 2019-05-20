import numpy as np

class Config(object):
	batch_size = 128
	embed_size = 80
	hidden_size = 80

	early_stopping = 20
	max_epochs = 256

	dropout = 0.9
	lr = 0.001
	l2 = 0.001

	word2vec_init = False
	embedding_init = np.sqrt(3)

	anneal_threshold = 1000
	anneal_by = 1.5

	cap_grads = False
	noisy_grads = False
	max_grad_val = 10

	num_attention_features = 4
	num_hops = 3

	max_allowed_inputs = 130
	num_train = 9000

	babi_id = "1"
	babi_test_id = ""

	floatX = np.float32

	train_mode= True
	strong_supervision =False