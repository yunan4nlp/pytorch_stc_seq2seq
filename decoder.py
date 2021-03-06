import torch.nn as nn
import torch.autograd
from read import Reader

class Decoder(nn.Module):
    def __init__(self, hyperParams):
        super(Decoder, self).__init__()
        self.hyperParams = hyperParams

        if hyperParams.wordEmbFile == "":
            self.wordEmb = nn.Embedding(hyperParams.responseWordNum, hyperParams.wordEmbSize)
            self.wordDim = hyperParams.wordEmbSize
        else:
            reader = Reader()
            self.wordEmb, self.wordDim = reader.load_pretrain(hyperParams.wordEmbFile, hyperParams.responseWordAlpha, hyperParams.unk)
        self.wordEmb.weight.requires_grad = hyperParams.wordFineTune
        self.dropOut = nn.Dropout(hyperParams.dropProb)
        self.gru = nn.GRU(input_size=hyperParams.rnnHiddenSize + self.wordDim,
                          hidden_size=hyperParams.rnnHiddenSize,
                          batch_first=True,
                          dropout=hyperParams.dropProb)

        self.linearLayer = nn.Linear(hyperParams.rnnHiddenSize, hyperParams.responseWordNum)
        self.softmax = nn.LogSoftmax()

    def initHidden(self, batch = 1):
        result = torch.autograd.Variable(torch.zeros(1, batch, self.hyperParams.rnnHiddenSize))
        return result

    def forward(self, encoderHidden, hidden, lastWordIndex):
        wordRepresent = self.wordEmb(lastWordIndex.squeeze())
        wordRepresent = self.dropOut(wordRepresent)
        wordRepresent = wordRepresent.unsqueeze(0)
        concat = torch.cat((encoderHidden, wordRepresent), 2)
        output, hidden = self.gru(concat.permute(1, 0, 2), hidden)
        output = torch.squeeze(output, 1)
        output = self.softmax(self.linearLayer(output))
        return output, hidden

