import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
from torchtext import data


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
        self.name = "recurrent"
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings_dropout = nn.Dropout(0.2)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=bidirectional, dropout=0.5)

        h0_tensor = torch.Tensor(1, hidden_dim)
        nn.init.xavier_normal_(h0_tensor, gain=1.)
        self.h_0_init = nn.Parameter(h0_tensor)

        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.orthogonal_(self.gru.__getattr__(p))

        if bidirectional:
            self.hidden2label = nn.Linear(2*hidden_dim, n_classes)
        else:
            self.hidden2label = nn.Linear(hidden_dim, n_classes)

    def init_hidden(self, batch_size : int) -> torch.Tensor:
        #h_0 = Variable(torch.zeros(2 if self.bidirectional else 1,
                       # batch_size, self.hidden_dim))
        h_0 = self.h_0_init.repeat(2 if self.bidirectional else 1, batch_size, 1)

        if torch.cuda.is_available():
            return h_0.cuda()
        else:
            return h_0

    def forward(self, sentence : Variable, lengths : torch.Tensor) -> torch.Tensor:
        batch_size = sentence.shape[0]
        x = self.embeddings(torch.transpose(sentence, 0, 1))  # time, batch, dim
        x = self.embeddings_dropout(x)
        packed_x = pack_padded_sequence(x, lengths)

        # Recurrent part
        hidden_in = self.init_hidden(batch_size)
        recurrent_out, hidden_out = self.gru(packed_x, hidden_in)
        recurrent_out, _ = pad_packed_sequence(recurrent_out)

        # Classification

        # FIXME: take final recurrent state or..
        # recurrent_out = torch.transpose(recurrent_out, 1, 0)  # batch, time, dim
        # dim = recurrent_out.size(2)
        # indices = lengths.view(-1, 1).unsqueeze(2).repeat(1, 1, dim) - 1
        # indices = indices.cuda() if torch.cuda.is_available() else indices
        # final_states = torch.squeeze(torch.gather(recurrent_out, 1, indices), dim=1)

        # FIXME: take hidden state
        # y = self.hidden2label(hidden_out.squeeze())
        recurrent_out = torch.transpose(recurrent_out, 1, 0)  # batch, time, dim
        y = self.hidden2label(recurrent_out)
        log_probs = F.log_softmax(y, 2)
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

class CNNRNN(nn.Module):
    def __init__(self, char_vocab_size, embed_size, word_vocab_size, n_classes, num_filters, kernel_size, n1, n2):
        super(CNNRNN, self).__init__()

        self.char_embedding = nn.Embedding(char_vocab_size, embed_size)
        self.conv1 = nn.Conv1d(embed_size, num_filters, kernel_size)
        self.relu = nn.ReLU()
        self.conv2_3 = nn.Conv1d(1, num_filters, kernel_size)
        self.conv2_4 = nn.Conv1d(1, num_filters, 4)
        self.conv2_5 = nn.Conv1d(1, num_filters, 5)
        self.dropout = nn.Dropout()
        self.lstm = nn.LSTM(3, 128, num_layers=1, bidirectional=True)
        self.linear_lstm = nn.Linear(128 * 2, n_classes)

        self.data_field = data.Field(pad_token = 0)

    def forward(self, sequence):
        embedded = self.char_embedding(sequence)
        bsz, seq_length, char_length, emb_dim = embedded.shape
        embedded = torch.transpose(embedded.view(bsz * seq_length, char_length, emb_dim), 1, 2)
        out = self.relu(self.conv1(embedded))
        out = self.dropout(out)

        out_3 = self.relu(self.conv2_3(out))
        out_4 = self.relu(self.conv2_4(out))
        out_5 = self.relu(self.conv2_5(out))

        maxpool_3 = nn.MaxPool1d(out_3.shape[2])
        maxpool_4 = nn.MaxPool1d(out_4.shape[2])
        maxpool_5 = nn.MaxPool1d(out_5.shape[2])
        y_3 = maxpool_3(out_3).squeeze(-1)
        y_4 = maxpool_4(out_4).squeeze(-1)
        y_5 = maxpool_5(out_5).squeeze(-1)

        y = torch.cat([y_3, y_4, y_5], 1)

        linear = nn.Linear(y.shape[-1], y.shape[-1])
        residual = linear(y)

        z = y + residual
        z = z.view(bsz, seq_length, -1)

        out, _ = self.lstm(z)
        output = self.linear_lstm(out)

        log_probs = F.log_softmax(output, 2)

        return log_probs
