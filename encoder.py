import torch.nn as nn
from read import Reader
import torch.autograd

class Encoder(nn.Module):
    def __init__(self, hyperParams):
        super(Encoder, self).__init__()
        self.hyperParams = hyperParams
        if hyperParams.wordEmbFile == "":
            self.wordEmb = nn.Embedding(hyperParams.postWordNum, hyperParams.wordEmbSize)
            self.wordDim = hyperParams.wordEmbSize
        else:
            reader = Reader()
            self.wordEmb, self.wordDim = reader.load_pretrain(hyperParams.wordEmbFile, hyperParams.postWordAlpha, hyperParams.unk)
        self.wordEmb.weight.requires_grad = hyperParams.wordFineTune
        self.dropOut = nn.Dropout(hyperParams.dropProb)
        self.gru = nn.GRU(input_size=self.wordDim,
                          hidden_size=hyperParams.rnnHiddenSize,
                          batch_first=True,
                          dropout=hyperParams.dropProb)



    def init_hidden(self):
        hidden = torch.autograd.Variable(torch.zeros(1, 1, self.hyperParams.rnnHiddenSize))
        return hidden

    def forward(self, postWordIndexes, hidden):
        wordRepresents = self.wordEmb(postWordIndexes)
        wordRepresents = self.dropOut(wordRepresents)
        output, hidden = self.gru(wordRepresents, hidden)
        return output, hidden

