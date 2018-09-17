import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math


class Model(nn.Module): # General class if we want all models to have common functions
    def __init__(self):
        super().__init__()


class RecurrentModel(nn.Module): # General recurrent class, because I want to create multiple recurrent models
    def __init__(self):
        super().__init__()


class GRUIdentifier(RecurrentModel):
    def __init__(self, vocab_size : int, n_classes : int, embedding_dim : int,
                 hidden_dim : int, bidirectional : bool, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=bidirectional)
        if bidirectional:
            self.hidden2label = nn.Linear(2*hidden_dim, n_classes)
        else:
            self.hidden2label = nn.Linear(hidden_dim, n_classes)            

    def init_hidden(self, batch_size : int) -> torch.Tensor:
        h_0 = Variable(torch.zeros(2 if self.bidirectional else 1, batch_size, self.hidden_dim))

        if torch.cuda.is_available():
            return h_0.cuda()
        else:
            return h_0

    def forward(self, sentence : Variable) -> torch.Tensor:
        batch_size = sentence.shape[0]
        x = self.embeddings(torch.transpose(sentence, 0, 1))
        hidden_in = self.init_hidden(batch_size)
        lstm_out, hidden_out = self.gru(x, hidden_in)
        y = self.hidden2label(lstm_out[-1])
        log_probs = F.log_softmax(y, 1)
        return log_probs


class CharModel(nn.Module):

  def __init__(self, n_chars, padding_idx, emb_dim=30, hidden_size=50, output_dim=50, dropout_p=0.5,
               bi=False):
    super(CharModel, self).__init__()

    self.input_dim = n_chars
    self.output_dim = output_dim
    self.dropout_p = dropout_p
    self.padding_idx = padding_idx
    self.hidden_size = hidden_size
    self.emb_dim = emb_dim

    self.embeddings = nn.Embedding(n_chars, emb_dim, padding_idx=padding_idx)
    self.init_embedding()
    self.char_emb_dropout = nn.Dropout(p=dropout_p)

    self.size = hidden_size * 2 if bi else hidden_size

  def init_embedding(self):
    init_range = math.sqrt(3 / self.emb_dim)
    embed = self.embeddings.weight.clone()
    embed.uniform_(-init_range, init_range)
    self.embeddings.weight.data.copy_(embed)

  def forward(self, sentence: Variable) -> torch.Tensor:

      # embed characters
      embedded = self.embeddings(sentence)

      # character model
      output = self.char_model(embedded)

      return output


class CharCNN(CharModel):

  def __init__(self, n_chars, padding_idx, emb_dim, num_filters, window_size, dropout_p, n_classes):

    super(CharCNN, self).__init__(n_chars, padding_idx, emb_dim=emb_dim, hidden_size=400, output_dim=100,
                                  dropout_p=dropout_p, bi=False)

    self.conv = nn.Conv1d(emb_dim, num_filters, window_size, padding=window_size - 1)
    self.xavier_uniform()
    self.hidden2label = nn.Linear(num_filters, n_classes)

  def xavier_uniform(self, gain=1.):

    # default pytorch initialization
    for name, weight in self.conv.named_parameters():
      if len(weight.size()) > 1:
          nn.init.xavier_uniform_(weight.data, gain=gain)
      elif "bias" in name:
        weight.data.fill_(0.)

  def char_model(self, embedded=None):

    embedded = torch.transpose(embedded, 1, 2)  # (bsz, dim, time)
    chars_conv = self.conv(embedded)
    chars = F.max_pool1d(chars_conv, kernel_size=chars_conv.size(2)).squeeze(2)
    labels = self.hidden2label(chars)
    log_probs = F.log_softmax(labels, 1)

    return log_probs