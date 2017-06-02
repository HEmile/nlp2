# first run a few imports:
import tensorflow as tf
import numpy as np
from pprint import pprint
import pickle
# check neuralibm1.py for the Model code
from neuralibm1_T2_concat import NeuralIBM1Model_T2
from neuralibm1_T3_gate import NeuralIBM1Model_T3
# check neuralibm1trainer.py for the Trainer code
from neuralibm1trainer import *
# the paths to our training and validation data, English side
train_e_path = 'data/training/hansards.36.2.e.gz'
train_f_path = 'data/training/hansards.36.2.f.gz'
dev_e_path = 'data/validation/dev.e.gz'
dev_f_path = 'data/validation/dev.f.gz'
dev_wa = 'data/validation/dev.wa.nonullalign'

# check utils.py if you want to see how smart_reader and bitext_reader work in detail
from utils import smart_reader, bitext_reader

# check vocabulary.py to see how the Vocabulary class is defined
from vocabulary import OrderedCounter, Vocabulary

# Using only 1000 words will result in many UNKs, but
# it will make training a lot faster.
# If you have a fast computer, a GPU, or a lot of time,
# try with 10000 instead.
max_tokens=1000

corpus_e = smart_reader(train_e_path)
vocabulary_e = Vocabulary(corpus=corpus_e, max_tokens=max_tokens)
pickle.dump(vocabulary_e, open("vocabulary_e.pkl", mode="wb"))
print("English vocabulary size: {}".format(len(vocabulary_e)))

corpus_f = smart_reader(train_f_path)
vocabulary_f = Vocabulary(corpus=corpus_f, max_tokens=max_tokens)
pickle.dump(vocabulary_f, open("vocabulary_f.pkl", mode="wb"))
print("French vocabulary size: {}".format(len(vocabulary_f)))
print()


tf.reset_default_graph()

with tf.Session() as sess:

  # some hyper-parameters
  # tweak them as you wish
  batch_size=128  # on CPU, use something much smaller e.g. 1-16
  max_length=30
  lr = 0.001
  lr_decay = 0.0  # set to 0.0 when using Adam optimizer (default)
  emb_dim = 64
  mlp_dim = 128

  # our model
  model = NeuralIBM1Model_T3(
    x_vocabulary=vocabulary_e, y_vocabulary=vocabulary_f,
    batch_size=batch_size, emb_dim=emb_dim, mlp_dim=mlp_dim, session=sess)

  # our trainer
  trainer = NeuralIBM1Trainer_T2_gate(
    model, train_e_path, train_f_path,
    dev_e_path, dev_f_path, dev_wa,
    num_epochs=10, batch_size=batch_size,
    max_length=max_length, lr=lr, lr_decay=lr_decay, session=sess)

  # now first TF needs to initialize all the variables
  print("Initializing variables..")
  sess.run(tf.global_variables_initializer())

  # now we can start training!
  print("Training started..")
  trainer.train()
