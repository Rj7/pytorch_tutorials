# coding: utf-8

from __future__ import unicode_literals, print_function, division

import math
import random
import re
import time
import unicodedata
import logging

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.nn.functional as F

from io import open
from torch import optim
from torch.autograd import Variable


FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO, filename='seq2seq_vanilla_RNN.log')
use_cuda = torch.cuda.is_available()
logging.info("GPU available: " + str(use_cuda))
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10
eng_prefixes = (
	"i am ", "i m ",
	"he is", "he s ",
	"she is", "she s",
	"you are", "you re ",
	"we are", "we re ",
	"they are", "they re "
)


class Lang:
	def __init__(self, name):
		self.name = name
		self.word2index = {}
		self.word2count = {}
		self.index2word = {0: "SOS", 1: "EOS"}
		self.n_words = 2  # Count SOS and EOS

	def addSentence(self, sentence):
		for word in sentence.split(' '):
			self.addWord(word)

	def addWord(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.n_words
			self.word2count[word] = 1
			self.index2word[self.n_words] = word
			self.n_words += 1
		else:
			self.word2count[word] += 1  # Turn a Unicode string to plain ASCII, thanks to


# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
	return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn'
	)


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
	s = unicodeToAscii(s.lower().strip())
	s = re.sub(r"([.!?])", r" \1", s)
	s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
	return s


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
	return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn'
	)


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
	s = unicodeToAscii(s.lower().strip())
	s = re.sub(r"([.!?])", r" \1", s)
	s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
	return s


def readLangs(lang1, lang2, reverse=False):
	logging.info("Reading lines...")

	# Read the file and split into lines
	lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')

	# Split every line into pairs and normalize
	pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

	# Reverse pairs, make Lang instances
	if reverse:
		pairs = [list(reversed(p)) for p in pairs]
		input_lang = Lang(lang2)
		output_lang = Lang(lang1)
	else:
		input_lang = Lang(lang1)
		output_lang = Lang(lang2)

	return input_lang, output_lang, pairs


def filterPair(p):
	return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH and p[1].startswith(eng_prefixes)


def filterPairs(pairs):
	return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):
	input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
	logging.info("Read %s sentence pairs" % len(pairs))
	pairs = filterPairs(pairs)
	logging.info("Trimmed to %s sentence pairs" % len(pairs))
	logging.info("Counting words...")
	for pair in pairs:
		input_lang.addSentence(pair[0])
		output_lang.addSentence(pair[1])
	logging.info("Counted words:")
	logging.info(input_lang.name + str(input_lang.n_words,))
	logging.info(output_lang.name + str(output_lang.n_words,))
	return input_lang, output_lang, pairs


class EncoderRNN(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(EncoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.embedding = nn.Embedding(input_size, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size)

	def forward(self, input, hidden):
		embedded = self.embedding(input).view(1, 1, -1)
		output = embedded
		output, hidden = self.gru(output, hidden)
		return output, hidden

	def initHidden(self):
		result = Variable(torch.zeros(1, 1, self.hidden_size))
		if use_cuda:
			return result.cuda()
		else:
			return result


class DecoderRNN(nn.Module):
	def __init__(self, hidden_size, output_size):
		super(DecoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.embedding = nn.Embedding(output_size, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size)
		self.out = nn.Linear(hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, input, hidden):
		output = self.embedding(input).view(1, 1, -1)
		output = F.relu(output)
		output, hidden = self.gru(output, hidden)
		output = self.softmax(self.out(output[0]))
		return output, hidden

	def initHidden(self):
		result = Variable(torch.zeros(1, 1, self.hidden_size))
		if use_cuda:
			return result.cuda()
		else:
			return result


def indexesFromSentence(lang, sentence):
	return [lang.word2index[word] for word in sentence.split(' ')]


def variableFromSentence(lang, sentence):
	indexes = indexesFromSentence(lang, sentence)
	indexes.append(EOS_token)
	result = Variable(torch.LongTensor(indexes).view(-1, 1))
	if use_cuda:
		return result.cuda()
	else:
		return result


def variablesFromPair(pair):
	input_variable = variableFromSentence(input_lang, pair[0])
	target_variable = variableFromSentence(output_lang, pair[1])
	return (input_variable, target_variable)


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
		  max_length=MAX_LENGTH):
	encoder_hidden = encoder.initHidden()

	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()

	input_length = input_variable.size()[0]
	target_length = target_variable.size()[0]

	loss = 0

	encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
	encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

	for ei in range(input_length):
		encoder_output, encoder_hidden = encoder(
			input_variable[ei], encoder_hidden)
		encoder_outputs[ei] = encoder_output[0][0]

	decoder_input = Variable(torch.LongTensor([[SOS_token]]))
	decoder_input = decoder_input.cuda() if use_cuda else decoder_input

	decoder_hidden = encoder_hidden

	for di in range(target_length):
		decoder_output, decoder_hidden = decoder(decoder_input,
												 decoder_hidden)
		topv, topi = decoder_output.data.topk(1)
		ni = topi[0][0]

		decoder_input = Variable(torch.LongTensor([[ni]]))
		decoder_input = decoder_input.cuda() if use_cuda else decoder_input

		loss += criterion(decoder_output, target_variable[di])
		if ni == EOS_token:
			break
	loss.backward()

	encoder_optimizer.step()
	decoder_optimizer.step()

	return loss.data[0] / target_length


def asMinutes(s):
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)


def timeSince(since, percent):
	now = time.time()
	s = now - since
	es = s / (percent)
	rs = es - s
	return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
	start = time.time()
	plot_losses = []
	print_loss_total = 0
	plot_loss_total = 0

	encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
	decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

	training_pairs = [variablesFromPair(random.choice(pairs)) for i in range(n_iters)]

	criterion = nn.NLLLoss()

	for iter in range(1, n_iters + 1):
		training_pair = training_pairs[iter - 1]
		input_variable = training_pair[0]
		target_variable = training_pair[1]

		loss = train(input_variable, target_variable,
					 encoder, decoder,
					 encoder_optimizer, decoder_optimizer,
					 criterion)
		print_loss_total += loss
		plot_loss_total += loss

		if iter % print_every == 0:
			print_loss_avg = print_loss_total / print_every
			print_loss_total = 0
			logging.info('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
										 iter, iter / n_iters * 100, print_loss_avg))

		if iter % plot_every == 0:
			plot_loss_avg = plot_loss_total / plot_every
			plot_losses.append(plot_loss_avg)
			plot_loss_total = 0


# showPlot(plot_losses)


def showPlot(points):
	plt.figure()
	fig, ax = plt.subplots()
	# this locator puts ticks at regular intervals
	loc = ticker.MultipleLocator(base=0.2)
	ax.yaxis.set_major_locator(loc)
	plt.plot(points)
	plt.show()


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
	input_variable = variableFromSentence(input_lang, sentence)
	logging.info(str(input_variable))
	input_length = input_variable.size()[0]
	encoder_hidden = encoder.initHidden()

	encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
	encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

	for ei in range(input_length):
		encoder_output, encoder_hidden = encoder(input_variable[ei],
												 encoder_hidden)

		encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

	decoder_input = Variable(torch.LongTensor([[SOS_token]]))
	decoder_input = decoder_input.cuda() if use_cuda else decoder_input

	decoder_hidden = encoder_hidden

	decoded_words = []
	for di in range(max_length):
		decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
		topv, topi = decoder_output.data.topk(1)
		ni = topi[0][0]
		if ni == EOS_token:
			decoded_words.append('<EOS>')
			break
		else:
			decoded_words.append(output_lang.index2word[ni])

		decoder_input = Variable(torch.LongTensor([[ni]]))
		decoder_input = decoder_input.cuda() if use_cuda else decoder_input

	return decoded_words


def evaluateRandomly(encoder, decoder, n=10):
	for i in range(n):
		pair = random.choice(pairs)
		logging.info('>', pair[0])
		logging.info('=', pair[1])
		output_words, attentions = evaluate(encoder, decoder, pair[0])
		output_sentence = ' '.join(output_words)
		logging.info('<', output_sentence)
		logging.info('')


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size)
decoder1 = DecoderRNN(hidden_size, output_lang.n_words)

if use_cuda:
	encoder1 = encoder1.cuda()
	decoder1 = decoder1.cuda()

trainIters(encoder1, decoder1, 75000, print_every=5000)

'''
CPU run on my computer
6m 1s (- 84m 27s) (5000 6%) 3.0857
12m 13s (- 79m 27s) (10000 13%) 2.5650
17m 54s (- 71m 39s) (15000 20%) 2.2152
23m 40s (- 65m 5s) (20000 26%) 1.9785
29m 25s (- 58m 50s) (25000 33%) 1.7381
35m 29s (- 53m 14s) (30000 40%) 1.5727
41m 9s (- 47m 2s) (35000 46%) 1.4157
46m 29s (- 40m 40s) (40000 53%) 1.2946
51m 50s (- 34m 33s) (45000 60%) 1.1534
57m 11s (- 28m 35s) (50000 66%) 1.0417
62m 32s (- 22m 44s) (55000 73%) 0.9519
67m 54s (- 16m 58s) (60000 80%) 0.8830
73m 13s (- 11m 15s) (65000 86%) 0.7526
78m 35s (- 5m 36s) (70000 93%) 0.7051
83m 59s (- 0m 0s) (75000 100%) 0.6603


GPU run on my computer
3m 57s (- 55m 31s) (5000 6%) 3.0508
7m 42s (- 50m 6s) (10000 13%) 2.5949
11m 31s (- 46m 5s) (15000 20%) 2.2525
15m 0s (- 41m 15s) (20000 26%) 1.9679
18m 30s (- 37m 1s) (25000 33%) 1.7611
21m 56s (- 32m 54s) (30000 40%) 1.5908
25m 16s (- 28m 53s) (35000 46%) 1.4057
28m 55s (- 25m 18s) (40000 53%) 1.2660
32m 59s (- 21m 59s) (45000 60%) 1.1556
37m 10s (- 18m 35s) (50000 66%) 1.0588
41m 22s (- 15m 2s) (55000 73%) 0.9506
45m 33s (- 11m 23s) (60000 80%) 0.8582
49m 45s (- 7m 39s) (65000 86%) 0.7872
54m 0s (- 3m 51s) (70000 93%) 0.7127
57m 37s (- 0m 0s) (75000 100%) 0.6593

'''
