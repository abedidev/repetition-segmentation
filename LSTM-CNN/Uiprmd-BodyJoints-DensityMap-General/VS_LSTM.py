#Built for Sequence to Sequence predictions
#To be used only for batchsize=1

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import signal
import sys
from os.path import exists
import math

def signal_handler(sig, frame):
    print('\nExiting...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

class LSTM(nn.Module):
    def __init__(self, nb_layers, nb_lstm_units, inputSize):
        super(LSTM, self).__init__()   
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda:1'
        self.nb_layers = nb_layers
        self.nb_lstm_units = nb_lstm_units
        self.input_size = inputSize
        self.scaling_factor = 1.0
        if exists("ScalingStatus.txt"):
            with open("ScalingStatus.txt", 'r') as ss:
                ss.readline()
                self.scaling_factor = float(ss.readline().split("\n")[0].split(" = ")[1])

        # build actual NN
        self.__build_model()
        
    def maxPool1dLout(self, Lin, kernel_size, stride=None, padding=0, dilation=1):
        if stride == None:
            stride = kernel_size
        #Lout = math.floor(((Lin+2*padding-dilation*(kernel_size-1)-1)/stride)+1)
        Lout = math.ceil((Lin-kernel_size+1)/stride)
        return Lout
    
    def conv1dLout(self, Lin, kernel_size, stride=1, padding=0, dilation=1):
        #Lout = math.floor(((Lin+2*padding-dilation*(kernel_size-1)-1)/stride)+1)
        Lout = math.ceil((Lin-kernel_size+1)/stride)
        return Lout

    def __build_model(self):
        # design LSTM
        self.lstm = nn.LSTM(
            input_size = self.input_size,
            hidden_size=self.nb_lstm_units,
            num_layers=self.nb_layers,
            batch_first=True,
        ).to(self.device)

        ## linear layer:
        #self.fully_connected = nn.Linear(self.nb_lstm_units, 1).to(self.device)
        #
        ##Sigmoid layer:
        #self.sigmoid = nn.Sigmoid().to(self.device)
        
        print("Dimensions: ",end="")
        Lin = self.nb_lstm_units
        print(str(Lin)+" -> ", end="")
        self.conv1 = nn.Conv1d(1,1,5).to(self.device)
        Lin = self.conv1dLout(Lin, 5)
        print(str(Lin)+" -> ", end="")
        #self.relu1 = nn.ReLU().to(self.device)
        self.linear1 = nn.Linear(Lin, 1).to(self.device)
        Lin = 1
        print(Lin,end="\n")
        self.softmax = nn.Softmax().to(self.device) 
        
        #Defining sequence of layers:
        self.sequence = nn.Sequential(
            self.conv1,
            #self.relu1,
            self.linear1,
            self.softmax
        )
        

    def init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.rand(self.nb_layers, self.batch_size, self.nb_lstm_units).to(self.device)
        hidden_b = torch.rand(self.nb_layers, self.batch_size, self.nb_lstm_units).to(self.device)

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    def getInputs(self, X, X_lengths):
        '''
        max_len = 0
        X_lengths = []
        num_features = 0
        for seq in X_list:
            X_lengths.append(len(seq))
            if len(seq) != 0:
                num_features = len(seq[0])
            if len(seq) > max_len:
                max_len = len(seq)
                
        padList = [0]*num_features
        
        for i in range(len(X_list)):
            iter = max_len - len(X_list[i])
            for j in range(iter):
                X_list[i].append(padList)
        
        X = torch.tensor(X_list, dtype = torch.float32)
        '''
        
        X = X.to(self.device)
        t = torch.tensor(X_lengths).to(self.device)
        sorted, indices = torch.sort(t, descending=True)
        X_lengths = [int(x) for x in sorted]
        X_copy = torch.clone(X)
        for ind1, ind2 in enumerate(indices):
            X[int(ind1)] = X_copy[int(ind2)]

        #print(X)
        #print(X_lengths)
        return [X, X_lengths, indices]

    def forward(self, X, X_lengths):
        #X, X_lengths, indices = self.getInputs(X_padded, X_lengths)
        self.batch_size, seq_len, _ = X.size()
        
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        
        self.hidden = self.init_hidden()

        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, nb_lstm_units)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        #out = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True)

        # now run through LSTM
        out, self.hidden = self.lstm(X, self.hidden)

        # undo the packing operation
        #out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        
        out = out.reshape(out.shape[1], 1, out.shape[2])
        out = self.sequence(out)*self.scaling_factor

        #out = self.fully_connected(out)
        #out = self.sigmoid(out)*self.scaling_factor
        #out = (self.sigmoid(out) * 40) + 10
        #out = (torch.tanh(out) * 20) + 30
        
        out = out.reshape(1, out.shape[0]) #Predicted density map
        
        #out_copy = torch.clone(out)
        #for ind1, ind2 in enumerate(indices):
        #    out[int(ind2)] = torch.clone(out_copy[int(ind1)])

        return out
    
def main():
    model = LSTM(4, 2*40, 40)
    #[1, 5, 30, -1]
    #X1 = torch.tensor([[[1, 20, 45, -1], [-1, 25, 50, 1], [1, 30, 55, -1], [-1, 35, 60, 1], [1, 20, 45, -1]]], dtype=torch.float32)
    X1 = torch.rand(1,500,40)
    X_lengths1 = [500]
    print(model(X1, X_lengths1))
    
if __name__ == "__main__":
    main()