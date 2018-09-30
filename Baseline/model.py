import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math


class MLP(nn.Module):
    """
     A multi-layer perceptron with pytorch modules
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        super(MLP, self).__init__()
        """
        n_inputs is the amount of inputs, n_hidden is a list of hidden layers
        containing the size of the layer and lastly the amount of classes
        """

        hidden_size = len(n_hidden)

        # Construct all the hidden parts based on the amount of hidden units
        self.hidden = nn.ModuleList()
        if hidden_size > 0:
            self.hidden.append(nn.Linear(n_inputs, n_hidden[0]))
            for i in range(hidden_size - 1):
                self.hidden.append(nn.Linear(n_hidden[i], n_hidden[i + 1]))
            self.hidden.append(nn.Linear(n_hidden[-1], n_classes))
        else:
            self.hidden.append(nn.Linear(n_inputs, n_classes))

        # The other two possible parts
        self.relu_layer = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        # Possibly add dropout for better outcomes
        #self.dropout = nn.Dropout()

    def forward(self, x):
      """
      Input: a tensor containing the input
      """

      out = x
      #out = self.dropout(out)

      num_hidd = len(self.hidden)

      if num_hidd == 1:
          out = self.hidden[0](x)
          out = self.softmax(out)
      else:
          for i, layer in enumerate(self.hidden):
              # For the last layer
              if (i + 1) == num_hidd:
                  out = layer(out)
                  out = self.softmax(out)
              # All the in between layers with relu activation
              else:
                  out = layer(out)
                  out = self.relu_layer(out)
      return out
