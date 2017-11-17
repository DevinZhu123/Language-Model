from data_preprocessing import *
import numpy as np
from numpy import linalg as LA
import math
import time


class lang_model:
    def __init__(self, hidden_dim, lr, batch_size, decay, path_to_train_set, path_to_val_set):
        self.lr = lr
        self.batch_size = batch_size
        self.decay = decay

        # load word-index map, and 4-grams from training set and val set.
        self.word_index = get_vocab_map(path_to_file=path_to_train_set)
        self.trainset = parse_4gram(word_index=self.word_index, path_to_file=path_to_train_set)
        self.valset = parse_4gram(word_index=self.word_index, path_to_file=path_to_val_set)
        print 'training set size: '+str(len(self.trainset))
        print 'val set size: ' + str(len(self.valset))

        # initialize weight matrix and embedding matrix.
        self.embedding_dim = 16
        self.embedding = np.zeros((8000, self.embedding_dim))
        self.hidden_dim = hidden_dim
        r1 = math.sqrt(6.0 / (self.embedding_dim * 3 + hidden_dim))
        r2 = math.sqrt(6.0 / (8000 + hidden_dim))
        self.embed_to_hid_weight = np.random.uniform(-r1, r1, (self.hidden_dim, self.embedding_dim * 3))
        self.embed_to_hid_bias = np.zeros((hidden_dim, 1))
        self.hid_to_output_weight = np.random.uniform(-r2, r2, (8000, self.hidden_dim))
        self.hid_to_output_bias = np.zeros((8000, 1))

        # initialize indicator matrix
        train_size = len(self.trainset)
        val_size = len(self.valset)
        self.train_indicator = np.zeros((8000, train_size))
        self.val_indicator = np.zeros((8000, val_size))
        for i in range(train_size):
            self.train_indicator[self.trainset[i][3], i] = 1
        for i in range(val_size):
            self.val_indicator[self.valset[i][3], i] = 1

    def linear_layer(self, input, w, b):
        """
        :param input: shape (input_dim, batch_size)
        :param w: shape (output_dim, input_dim)
        :param b: shape (output_size, 1)
        :return: output shape (output_dim, batch_size)
        """
        return np.dot(w, input) + b

    def linear_layer_bacward(self, grad, w, input):
        """

        :param grad: the same as output tensor shape, i.e. (output_dim, batch_size)
        :param w: shape (output_dim, input_dim)
        :param input: shape (input_dim, batch_size)
        :return:
            grad_w: shape (output_dim, input_dim)
            grad_b: (output_dim, 1)
            grad_h: shape (input_dim, batch_size)
        """
        # print grad.shape, input.shape
        grad_w = np.dot(grad, np.transpose(input))
        grad_b = np.sum(grad, axis=1).reshape((grad.shape[0], 1))
        grad_h = np.dot(np.transpose(w), grad)
        return {'w': grad_w, 'b': grad_b, 'h': grad_h}

    def softmax_forward(self, input, w, b):
        """
        This layer includes a linear combination and a softmax layer.
        :param input: shape (input_dim, batch_size)
        :param w: shape (output_dim, input_dim)
        :param b: shape (output_dim, 1)
        :return: output shape (output_dim, batch_size)
        """
        a = np.dot(w, input) + b
        exp_a = np.exp(a)
        output = exp_a / np.sum(exp_a, axis=0).reshape((1, a.shape[1]))
        return output

    def softmax_backward(self, indicator, w, input, output):
        """

        :param indicator: shape (vacab_size, batch_size)
        :param w: shape (output_dim, input_dim)
        :param input: shape ()
        :param output:
        :return:
        """
        grad_a = output - indicator
        grad_w = np.dot(grad_a, np.transpose(input))
        grad_b = np.sum(grad_a, axis=1).reshape((grad_a.shape[0], 1))
        grad_h = np.dot(np.transpose(w), grad_a)
        return {'w': grad_w, 'b': grad_b, 'h': grad_h}

    def cal_cross_entropy(self, indicator, softmax):
        # ret = np.trace(-np.log(np.dot(np.transpose(indicator), softmax)))
        # ret = np.sum(np.log(softmax) * indicator)
        ret = np.sum(-np.log(softmax[indicator==1]))
        return ret

    def update_params(self, hidden_grad, softmax_grad):
        self.hid_to_output_weight -= (softmax_grad['w'] / self.batch_size + self.decay * LA.norm(softmax_grad['w'])) * self.lr
        self.hid_to_output_bias -= (softmax_grad['b'] / self.batch_size) * self.lr
        self.embed_to_hid_weight -= (hidden_grad['w'] / self.batch_size + self.decay * LA.norm(hidden_grad['w'])) * self.lr
        self.embed_to_hid_bias -= (hidden_grad['b'] / self.batch_size) * self.lr

    def model(self, datatype, task):
        benchmark = False
        if datatype == 'trainset':
            dataset = self.trainset
            indicator = self.train_indicator
        else:
            dataset = self.valset
            indicator = self.val_indicator

        num_batch = len(dataset) / self.batch_size
        entropy_errors = np.zeros(num_batch)  # record entropy errors in each batch
        for i in range(num_batch):
            start_time = time.time()
            train_batch = np.zeros((self.embedding_dim * 3, self.batch_size))
            # indicator = np.zeros((8000, self.batch_size))

            # construct the training matrix and indicator matrix for a batch
            for j in range(self.batch_size):
                sample = dataset[i*self.batch_size+j]
                # indicator[sample[3], j] = 1
                train_batch[:, j] = self.embedding[[sample[0], sample[1], sample[2]]].reshape((self.embedding_dim*3))
            if benchmark:
                init_time = time.time()
                print 'initialization time: {}'.format(init_time-start_time)
            # forward
            hidden_before = time.time()
            hidden_output = self.linear_layer(input=train_batch,
                                              w=self.embed_to_hid_weight,
                                              b=self.embed_to_hid_bias)
            if benchmark:
                soft_before = time.time()
                print 'hidden time: {}'.format(soft_before - hidden_before)
            softmax_output = self.softmax_forward(input=hidden_output,
                                                  w=self.hid_to_output_weight,
                                                  b=self.hid_to_output_bias)
            if benchmark:
                soft_end = time.time()
                print 'softmax time: {}'.format(soft_end - soft_before)
            entropy_errors[i] = self.cal_cross_entropy(indicator=indicator[:, i*self.batch_size:(i+1)*self.batch_size],
                                                       softmax=softmax_output)
            if benchmark:
                forward_time = time.time()
                print 'cal entropy time: {}'.format(forward_time - soft_end)
                print 'forward time: {}'.format(forward_time-init_time)

            # backward
            if task != 'train':
                continue
            softmax_grad = self.softmax_backward(indicator=indicator[:, i*self.batch_size:(i+1)*self.batch_size],
                                                 w=self.hid_to_output_weight,
                                                 input=hidden_output,
                                                 output=softmax_output)
            hidden_grad = self.linear_layer_bacward(grad=softmax_grad['h'],
                                                    w=self.embed_to_hid_weight,
                                                    input=train_batch)
            self.update_params(hidden_grad=hidden_grad,
                               softmax_grad=softmax_grad)

            # update word embedding matrix
            for j in range(self.batch_size):
                sample = dataset[i * self.batch_size + j]
                self.embedding[sample[0]] -= self.lr * hidden_grad['h'][:self.embedding_dim, j]
                self.embedding[sample[1]] -= self.lr * hidden_grad['h'][self.embedding_dim:self.embedding_dim * 2, j]
                self.embedding[sample[2]] -= self.lr * hidden_grad['h'][self.embedding_dim * 2:self.embedding_dim * 3, j]

        cross_entropy_error = np.sum(entropy_errors) / num_batch / self.batch_size
        return cross_entropy_error

    def run_model(self, num_epoch=100):
        train_errors = np.zeros(num_epoch)
        val_errors = np.zeros(num_epoch)
        for i in range(num_epoch):
            self.model(datatype='trainset', task='train')
            # train_errors[i] = self.model(datatype='trainset', task='val')
            val_errors[i] = self.model(datatype='valset', task='val')
            print 'epoch '+str(i)+'\ttraining error: '+str(train_errors[i])+'\tvalid error: '+str(np.exp(val_errors[i]))
        return train_errors, val_errors


lm = lang_model(hidden_dim=128,
                lr=0.1,
                batch_size=256,
                decay=0,
                path_to_train_set='./data/train.txt',
                path_to_val_set='./data/val.txt')
lm.run_model()
