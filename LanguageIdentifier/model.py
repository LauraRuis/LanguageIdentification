import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


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