import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
from torchtext import data

from CodeSwitching.utils import PAD_TOKEN

class Model(nn.Module): # General class if we want all models to have common functions
    def __init__(self):
        super().__init__()


class RecurrentModel(nn.Module): # General recurrent class, because I want to create multiple recurrent models
    def __init__(self):
        super().__init__()


class GRUIdentifier(RecurrentModel):
    def __init__(self, vocab_size : int, n_classes : int, embedding_dim : int,
                 hidden_dim : int, bidirectional : bool, vocab, **kwargs):
        super().__init__()
        self.name = "recurrent"
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings_dropout = nn.Dropout(0.2)
        self.n_classes = n_classes
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=bidirectional, dropout=0.5)

        h0_tensor = torch.Tensor(1, hidden_dim)
        nn.init.xavier_normal_(h0_tensor, gain=1.)
        self.h_0_init = nn.Parameter(h0_tensor)
        self.vocab = vocab

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

    def __init__(self, n_chars, padding_idx, emb_dim=30, output_dim=50, dropout_p=0.5, embed_chars=True):
        super(CharModel, self).__init__()

        self.name = "convolutional"

        self.input_dim = n_chars
        self.output_dim = output_dim
        self.dropout_p = dropout_p
        self.padding_idx = padding_idx
        self.emb_dim = emb_dim
        self.embed_chars = embed_chars

        if embed_chars:
            self.embeddings = nn.Embedding(n_chars, emb_dim, padding_idx=padding_idx)
            self.init_embedding()

        self.char_emb_dropout = nn.Dropout(p=dropout_p)

    def init_embedding(self):
        init_range = math.sqrt(3 / self.emb_dim)
        embed = self.embeddings.weight.clone()
        embed.uniform_(-init_range, init_range)
        self.embeddings.weight.data.copy_(embed)

    def forward(self, sentence: Variable=None, lengths: torch.Tensor=None) -> torch.Tensor:
        # embed characters
        if self.embed_chars:
            embedded = self.embeddings(sentence)
            embedded = self.char_emb_dropout(embedded)
        else:
            embedded = sentence

        # character model
        output = self.char_model(embedded, lengths)
        return output


class SmallCNN(CharModel):

    def __init__(self, n_chars, padding_idx, emb_dim, num_filters, window_size, dropout_p, n_classes):
        super(SmallCNN, self).__init__(n_chars, padding_idx, emb_dim=emb_dim,  output_dim=100, dropout_p=dropout_p,
                                       embed_chars=True)

        self.conv1 = nn.Conv1d(emb_dim, num_filters, 5, padding=5 - 1)
        self.conv2 = nn.Conv1d(num_filters, num_filters, window_size, padding=window_size - 1)
        self.xavier_uniform(name="conv1")
        self.xavier_uniform(name="conv2")
        self.relu = nn.ReLU()

        self.hidden_dim = 128
        self.chars2hidden = nn.Linear(num_filters, self.hidden_dim)
        self.lstm = nn.LSTM(emb_dim + num_filters, self.hidden_dim, num_layers=1, bidirectional=False)
        self.hidden2label = nn.Linear(self.hidden_dim, n_classes)
        self.n_classes = n_classes

    def xavier_uniform(self, name="", gain=1.):

        # default pytorch initialization
        pars = getattr(self, name)
        for name, weight in pars.named_parameters():
            if len(weight.size()) > 1:
                nn.init.xavier_uniform_(weight.data, gain=gain)
            elif "bias" in name:
                weight.data.fill_(0.)

    def init_hidden(self, batch_size : int) -> torch.Tensor:
        h_0 = torch.Tensor(1, self.hidden_dim).repeat(1, batch_size, 1)

        if torch.cuda.is_available():
            return h_0.cuda()
        else:
            return h_0

    def char_model(self, embedded: torch.Tensor=None, lengths: torch.Tensor=None) -> torch.Tensor:
        embedded = torch.transpose(embedded, 1, 2)  # (bsz, dim, time)
        if not self.train:
            print("Time: {}".format(embedded.shape[2]))
        chars_conv = self.conv1(embedded)
        chars_conv = self.relu(self.conv2(chars_conv))
        chars = F.max_pool1d(chars_conv, kernel_size=chars_conv.size(2)).squeeze(2)

        embedded = embedded.transpose(1, 2)  # batch, time, dim
        embeddings_conv = torch.cat((embedded, chars.unsqueeze(1).repeat(1, embedded.shape[1], 1)), 2)
        packed_embeddings = pack_padded_sequence(embeddings_conv.transpose(0, 1), lengths)

        output, _ = self.lstm(packed_embeddings)
        output, _ = pad_packed_sequence(output)
        output = output.transpose(0, 1)
        labels = self.hidden2label(output)
        log_probs = F.log_softmax(labels, 2)

        return log_probs


class CNNRNN(nn.Module):
    def __init__(self, char_vocab_size, embed_size, word_vocab_size, n_classes, num_filters, kernel_size, n1, n2, vocab):
        super(CNNRNN, self).__init__()

        self.char_embedding = nn.Embedding(char_vocab_size, embed_size, padding_idx=1)
        self.conv1 = nn.Conv1d(embed_size, n1, kernel_size)
        self.relu = nn.ReLU()
        self.conv2_3 = nn.Conv1d(n1, num_filters, kernel_size)
        self.conv2_4 = nn.Conv1d(n1, num_filters, 4)
        self.conv2_5 = nn.Conv1d(n1, num_filters, 5)
        self.dropout = nn.Dropout(p=0.25)

        self.lstm = nn.LSTM(3 * num_filters, 128, num_layers=1, bidirectional=True)
        self.n_classes = n_classes


        self.hidden_dim = 128
        self.bidirectional = True
        # self.lstm = nn.LSTM(embed_size, self.hidden_dim, num_layers=1, bidirectional=self.bidirectional)
        self.linear_lstm = nn.Linear(128 * 2, n_classes)

        self.vocab = vocab
        self.name = "cnnrnn"
        self.linear = nn.Linear(3 * num_filters, 3 * num_filters)

    def init_hidden(self, batch_size : int) -> torch.Tensor:
        #h_0 = Variable(torch.zeros(2 if self.bidirectional else 1,
                       # batch_size, self.hidden_dim))
        h_0 = torch.Tensor(1, self.hidden_dim).repeat(2 if self.bidirectional else 1, batch_size, 1)

        if torch.cuda.is_available():
            return h_0.cuda()
        else:
            return h_0

    def forward(self, sequence : Variable, char_lengths : torch.Tensor, lengths : torch.Tensor) -> torch.Tensor:

        bsz, seq_length, char_length = sequence.shape

        embedded = self.char_embedding(sequence)
        embedded = embedded.transpose(2, 1)
        out = self.relu(self.conv1(embedded))

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

        residual = self.linear(y)

        z = y + self.relu(residual)


        #z = torch.cat(word_reps, 1)
        # z = self.dropout(z)
        packed_embedded = pack_padded_sequence(z.transpose(0, 1), lengths)

        recurrent_out, _ = self.lstm(packed_embedded)

        recurrent_out, _ = pad_packed_sequence(recurrent_out)

        output = self.linear_lstm(recurrent_out.transpose(0, 1))

        log_probs = F.log_softmax(output, 2)

        return log_probs
