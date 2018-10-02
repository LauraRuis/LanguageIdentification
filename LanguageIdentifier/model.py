import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math

from LanguageIdentifier.utils import PAD_TOKEN


class Model(nn.Module):  # General class if we want all models to have common functions
    def __init__(self):
        super().__init__()


class RecurrentModel(nn.Module):  # General recurrent class, because I want to create multiple recurrent models
    def __init__(self):
        super().__init__()


class GRUIdentifier(RecurrentModel):
    def __init__(self, vocab_size : int, n_classes : int, embedding_dim : int,
                 hidden_dim : int, bidirectional : bool, dropout_p : float,
                 **kwargs):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim,
                          bidirectional=bidirectional, batch_first=True)
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
        # Initialise hidden state with learned hidden state
        h_0 = self.h_0_init.repeat(2 if self.bidirectional else 1, batch_size, 1)

        if torch.cuda.is_available():
            return h_0.cuda()
        else:
            return h_0

    def forward(self, sentence : Variable, lengths : torch.Tensor) -> torch.Tensor:

        batch_size = sentence.shape[0]
        x = self.embeddings(sentence)  # batch, time, dim
        x = self.dropout(x)
        packed_x = pack_padded_sequence(x, lengths, batch_first=True)

        # Recurrent part
        hidden_in = self.init_hidden(batch_size)
        recurrent_out, hidden_out = self.gru(packed_x, hidden_in)
        recurrent_out, _ = pad_packed_sequence(recurrent_out, batch_first=True)

        # Unpack packed sequences
        dim = recurrent_out.size(2)
        indices = lengths.view(-1, 1).unsqueeze(2).repeat(1, 1, dim) - 1
        indices = indices.cuda() if torch.cuda.is_available() else indices
        final_states = torch.squeeze(torch.gather(recurrent_out, 1, indices), dim=1)

        # Classification
        y = self.hidden2label(final_states)

        if len(y.shape) == 1:
            y = y.unsqueeze(0)
        log_probs = F.log_softmax(y, 1)
        return log_probs


class CharModel(nn.Module):

    def __init__(self, n_chars, padding_idx, emb_dim=30, output_dim=50, dropout_p=0.5, embed_chars=True):
        super(CharModel, self).__init__()

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

    def forward(self, sentence: Variable) -> torch.Tensor:
        # embed characters
        if self.embed_chars:
            embedded = self.embeddings(sentence)
            embedded = self.char_emb_dropout(embedded)
        else:
            embedded = sentence

        # character model
        output = self.char_model(embedded)
        return output


class SmallCNN(CharModel):

    def __init__(self, n_chars, padding_idx, emb_dim, num_filters, window_size, dropout_p, n_classes):
        super(SmallCNN, self).__init__(n_chars, padding_idx, emb_dim=emb_dim,  output_dim=100, dropout_p=dropout_p,
                                       embed_chars=True)

        self.conv1 = nn.Conv1d(emb_dim, num_filters, window_size, padding=window_size - 1)
        self.conv2 = nn.Conv1d(num_filters, num_filters, window_size, padding=window_size - 1)
        self.xavier_uniform(name="conv1")
        self.xavier_uniform(name="conv2")
        self.hidden2label = nn.Linear(num_filters, n_classes)

    def xavier_uniform(self, name="", gain=1.):

        # default pytorch initialization
        pars = getattr(self, name)
        for name, weight in pars.named_parameters():
            if len(weight.size()) > 1:
                nn.init.xavier_uniform_(weight.data, gain=gain)
            elif "bias" in name:
                weight.data.fill_(0.)

    def char_model(self, embedded=None):
        embedded = torch.transpose(embedded, 1, 2)  # (bsz, dim, time)
        if not self.train:
            print("Time: {}".format(embedded.shape[2]))
        chars_conv = self.conv1(embedded)
        chars_conv = self.conv2(chars_conv)
        chars = F.max_pool1d(chars_conv, kernel_size=chars_conv.size(2)).squeeze(2)
        labels = self.hidden2label(chars)
        log_probs = F.log_softmax(labels, 1)

        return log_probs


class CharCNN(CharModel):

    def __init__(self, n_chars, padding_idx, emb_dim, dropout_p, n_classes, length):
        super(CharCNN, self).__init__(n_chars, padding_idx, emb_dim=emb_dim, output_dim=100,
                                      dropout_p=dropout_p, embed_chars=False)

        self.n_chars = n_chars
        # in_channels, out_channels, kernel_size, stride, padding
        conv_stride = 1
        max_pool_kernel_size = 3
        max_pool_stride = 3
        padding = 0
        conv_spec_1 = dict(in_channels=n_chars, out_channels=256, kernel_size=7, padding=0)
        conv_spec_2 = dict(in_channels=256, out_channels=256, kernel_size=7, padding=0)
        conv_spec_3 = dict(in_channels=256, out_channels=256, kernel_size=3, padding=0)
        conv_spec_4 = dict(in_channels=256, out_channels=256, kernel_size=3, padding=0)
        conv_spec_5 = dict(in_channels=256, out_channels=256, kernel_size=3, padding=0)
        conv_spec_6 = dict(in_channels=256, out_channels=256, kernel_size=3, padding=0)
        network = [conv_spec_1, 'MaxPool', conv_spec_2, 'MaxPool', conv_spec_3,
                   conv_spec_4, conv_spec_5, conv_spec_6, 'MaxPool']

        layers = []
        for layer in network:
            if layer == 'MaxPool':
                layers.append(nn.MaxPool1d(kernel_size=max_pool_kernel_size, stride=max_pool_stride, padding=padding))
            else:
                conv = nn.Conv1d(layer['in_channels'], layer['out_channels'],
                                 kernel_size=layer['kernel_size'], stride=conv_stride, padding=layer['padding'])
                relu = nn.ReLU(inplace=True)
                layers.extend([conv, relu])

        self.layers = nn.Sequential(*layers)
        self.fc1 = nn.Linear(int((length - 96)/27) * 256, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.classifier = nn.Linear(1024, n_classes)
        self.gaussian_init()

    def gaussian_init(self, mean=0., std=0.05):
        for name, weight in self.named_parameters():
            if len(weight.size()) > 1:
                nn.init.normal_(weight.data, mean=mean, std=std)
            elif "bias" in name:
                weight.data.fill_(0.)

    def xavier_uniform(self, gain=1.):
        # default pytorch initialization
        for name, weight in self.named_parameters():
          if len(weight.size()) > 1:
              nn.init.xavier_uniform_(weight.data, gain=gain)
          elif "bias" in name:
            weight.data.fill_(0.)

    def one_hot(self, data):
        batch_size, time = data.shape
        flattened = data.view(-1).unsqueeze(1)

        # One hot encoding buffer that you create out of the loop and just keep reusing
        y_onehot = torch.FloatTensor(batch_size * time, self.n_chars)
        y_onehot = y_onehot.cuda() if torch.cuda.is_available() else y_onehot

        # In your for loop
        y_onehot.zero_()
        y_onehot.scatter_(1, flattened, 1)
        return y_onehot.view(batch_size, time, self.n_chars)

    def char_model(self, embedded=None):

        # one hot vector
        embedded = self.one_hot(embedded)
        embedded = torch.transpose(embedded, 1, 2)  # (bsz, dim, time)

        # conv net
        bsz = embedded.shape[0]
        chars_conv = self.layers(embedded)

        # fully connected layers
        output = self.fc1(chars_conv.view(bsz, -1))
        output = self.char_emb_dropout(output)
        output = self.fc2(output)

        # dropout and classify
        output = self.char_emb_dropout(output)
        labels = self.classifier(output)

        # softmax
        log_probs = F.log_softmax(labels, 1)
        return log_probs


class CNNRNN(nn.Module):
    def __init__(self, char_vocab_size, embed_size, n_classes, num_filters, kernel_size, n1, vocab):
        super(CNNRNN, self).__init__()

        self.char_embedding = nn.Embedding(char_vocab_size, embed_size, padding_idx=1)
        self.conv1 = nn.Conv1d(embed_size, n1, kernel_size)
        self.relu = nn.ReLU()
        self.conv2_3 = nn.Conv1d(n1, num_filters, kernel_size)
        self.conv2_4 = nn.Conv1d(n1, num_filters, 4)
        self.conv2_5 = nn.Conv1d(n1, num_filters, 5)
        self.dropout = nn.Dropout(p=0.25)
        self.lstm = nn.LSTM(3 * num_filters, 128, num_layers=1, bidirectional=True)  # TODO fix back

        self.hidden_dim = 128
        self.bidirectional = True
        self.linear_lstm = nn.Linear(128 * 2, n_classes)

        self.vocab = vocab
        self.name = "cnnrnn"
        self.linear = nn.Linear(3 * num_filters, 3 * num_filters)

    def init_hidden(self, batch_size : int) -> torch.Tensor:
        h_0 = torch.Tensor(1, self.hidden_dim).repeat(2 if self.bidirectional else 1, batch_size, 1)

        if torch.cuda.is_available():
            return h_0.cuda()
        else:
            return h_0

    def forward(self, sequence : Variable, char_lengths : torch.Tensor, lengths : torch.Tensor) -> torch.Tensor:

        bsz, seq_length, char_length = sequence.shape

        word_reps = []
        for t in range(seq_length):

            words = sequence[:, t]
            max_length = char_lengths[:, t].max()
            max_length = max_length if max_length > 7 else 8
            words_chopped = words[:, :max_length]

            words_chopped = self.char_embedding(words_chopped)
            embedded = torch.transpose(words_chopped, 1, 2)
            out = self.relu(self.conv1(embedded))
            # out = self.dropout(out)

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
            word_reps.append(z.unsqueeze(1))

        z = torch.cat(word_reps, 1)
        # z = self.dropout(z)
        packed_embedded = pack_padded_sequence(z.transpose(0, 1), lengths)

        recurrent_out, (hidden_states, cell_states) = self.lstm(packed_embedded)

        cell_states = cell_states.transpose(0, 1).contiguous().view(cell_states.shape[1], -1)

        output = self.linear_lstm(cell_states)

        log_probs = F.log_softmax(output, 1)

        return log_probs
