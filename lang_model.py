from data_preprocessing import *
import numpy as np
import math


class lang_model:
    def __init__(self, hidden_dim, path_to_train_set, path_to_val_set):
        # load word-index map, and 4-grams from training set and val set.
        self.word_index = get_vocab_map(path_to_file=path_to_train_set)
        self.trainset = parse_4gram(word_index=self.word_index, path_to_file=path_to_train_set)
        self.valset = parse_4gram(word_index=self.word_index, path_to_file=path_to_val_set)

        # initialize weight matrix.
        self.embedding_dim = 16
        self.hidden_dim = hidden_dim
        r1 = math.sqrt(6.0 / (self.embedding_dim * 3 + hidden_dim))
        r2 = math.sqrt(6.0 / (8000 + hidden_dim))
        self.embed_to_hid_weight = np.random.uniform(-r1, r1, (self.hidden_dim, self.embedding_dim * 3))
        self.embed_to_hid_bias = 0
        self.hid_to_output_weight = np.random.uniform(-r2, r2, (8000, self.hidden_dim))
        self.hid_to_output_bias = 0

    def linear_layer(self, input, w, b):
        return np.dot(w, input) + b

    def linear_layer_bacward(self, grad, w, input, output):
        grad_w = np.dot(grad, np.transpose(input))
        grad_b = np.sum(grad, axis=1).reshape((grad.shape[0], 1))
        grad_h = np.dot(np.transpose(w), grad)
        return {'w': grad_w, 'b': grad_b, 'h': grad_h}

    def softmax_forward(self, input, w, b):
        a = np.dot(w, input) + b
        output = np.exp(a) / np.sum(np.exp(a), axis=0).reshape((1, a.shape[1]))
        return output

    def softmax_backward(self, indicator, w, input, output):
        grad_a = output - indicator
        grad_w = np.dot(grad_a, np.transpose(input))
        grad_b = np.sum(grad_a, axis=1).reshape((grad_a.shape[0], 1))
        grad_h = np.dot(np.transpose(w), grad_a)
        return {'w': grad_w, 'b': grad_b, 'h': grad_h}

    def model(self, task, batch_size, lr, ):




    def run_model(self, lr, batch_size, num_epoch=100):
        for i in range(num_epoch):

