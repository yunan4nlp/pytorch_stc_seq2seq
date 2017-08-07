from read import Reader
from hyperParams import  HyperParams
from optparse import OptionParser
from instance import  Example
from encoder import  Encoder
from decoder import  Decoder
import torch.nn
import torch.autograd
import torch.nn.functional
import random

class Trainer:
    def __init__(self):
        self.post_word_state = {}
        self.response_word_state = {}

        self.hyperParams = HyperParams()

    def createAlphabet(self, trainInsts):
        print("create alpha.................")
        for inst in trainInsts:
            for w in inst.post:
                if w not in self.post_word_state:
                    self.post_word_state[w] = 1
                else:
                    self.post_word_state[w] += 1

            for w in inst.response:
                if w not in self.response_word_state:
                    self.response_word_state[w] = 1
                else:
                    self.response_word_state[w] += 1


        self.post_word_state[self.hyperParams.unk] = self.hyperParams.wordCutOff + 1
        self.post_word_state[self.hyperParams.padding] = self.hyperParams.wordCutOff + 1
        self.post_word_state[self.hyperParams.start] = self.hyperParams.wordCutOff + 1
        self.post_word_state[self.hyperParams.end] = self.hyperParams.wordCutOff + 1

        self.hyperParams.postWordAlpha.initial(self.post_word_state, self.hyperParams.wordCutOff)
        self.hyperParams.postWordAlpha.set_fixed_flag(True)

        self.hyperParams.postWordNum = self.hyperParams.postWordAlpha.m_size
        self.hyperParams.postUnkWordID = self.hyperParams.postWordAlpha.from_string(self.hyperParams.unk)
        self.hyperParams.postPaddingID = self.hyperParams.postWordAlpha.from_string(self.hyperParams.padding)
        self.hyperParams.postStartID = self.hyperParams.postWordAlpha.from_string(self.hyperParams.start)
        self.hyperParams.postEndID = self.hyperParams.postWordAlpha.from_string(self.hyperParams.end)

        self.response_word_state[self.hyperParams.unk] = self.hyperParams.wordCutOff + 1
        self.response_word_state[self.hyperParams.padding] = self.hyperParams.wordCutOff + 1
        self.response_word_state[self.hyperParams.start] = self.hyperParams.wordCutOff + 1
        self.response_word_state[self.hyperParams.end] = self.hyperParams.wordCutOff + 1

        self.hyperParams.responseWordAlpha.initial(self.response_word_state, self.hyperParams.wordCutOff)
        self.hyperParams.responseWordAlpha.set_fixed_flag(True)

        self.hyperParams.responseWordNum = self.hyperParams.responseWordAlpha.m_size
        self.hyperParams.responseUnkWordID = self.hyperParams.responseWordAlpha.from_string(self.hyperParams.unk)
        self.hyperParams.responsePaddingID = self.hyperParams.responseWordAlpha.from_string(self.hyperParams.padding)
        self.hyperParams.responseStartID = self.hyperParams.responseWordAlpha.from_string(self.hyperParams.start)
        self.hyperParams.responseEndID = self.hyperParams.responseWordAlpha.from_string(self.hyperParams.end)

        print("Post Word num: ", self.hyperParams.postWordNum)
        print("Post UNK ID: ", self.hyperParams.postUnkWordID)
        print("Post Padding ID: ", self.hyperParams.postPaddingID)
        print("Post Start num: ", self.hyperParams.postStartID)
        print("Post End num: ", self.hyperParams.postEndID)
        print('====')
        print("Response Word num: ", self.hyperParams.responseWordNum)
        print("Response UNK ID: ", self.hyperParams.responseUnkWordID)
        print("Response Padding ID: ", self.hyperParams.responsePaddingID)
        print("Response Start num: ", self.hyperParams.responseStartID)
        print("Response End num: ", self.hyperParams.responseEndID)

    def instance2Example(self, insts):
        exams = []
        for inst in insts:
            example = Example()

            example.postLen = len(inst.post)
            example.postWordIndexs = torch.autograd.Variable(torch.LongTensor(1, example.postLen + 2))
            example.postWordIndexs.data[0][0] = self.hyperParams.postStartID
            for idx in range(example.postLen):
                w = inst.post[idx]
                wordId = self.hyperParams.postWordAlpha.from_string(w)
                if wordId == -1:
                    wordId = self.hyperParams.postUnkWordID
                example.postWordIndexs.data[0][idx + 1] = wordId
            example.postWordIndexs.data[0][example.postLen + 1] = self.hyperParams.postEndID

            example.responseLen = len(inst.response)
            example.responseWordIndexs = torch.autograd.Variable(torch.LongTensor(1, example.responseLen + 1))
            for idx in range(example.responseLen):
                w = inst.response[idx]
                wordId = self.hyperParams.responseWordAlpha.from_string(w)
                if wordId == -1:
                    wordId = self.hyperParams.responseUnkWordID
                example.responseWordIndexs.data[0][idx] = wordId
            example.responseWordIndexs.data[0][example.responseLen] = self.hyperParams.responseEndID

            exams.append(example)
        return exams

    def train(self, train_file, dev_file, test_file):
        self.hyperParams.show()
        torch.set_num_threads(self.hyperParams.thread)
        reader = Reader()

        trainInsts = reader.readInstances(train_file, self.hyperParams.maxInstance)
        devInsts = reader.readInstances(dev_file, self.hyperParams.maxInstance)
        testInsts = reader.readInstances(test_file, self.hyperParams.maxInstance)

        print("Training Instance: ", len(trainInsts))
        print("Dev Instance: ", len(devInsts))
        print("Test Instance: ", len(testInsts))

        self.createAlphabet(trainInsts)

        trainExamples = self.instance2Example(trainInsts)
        devExamples = self.instance2Example(devInsts)
        testExamples = self.instance2Example(testInsts)

        self.encoder = Encoder(self.hyperParams)
        self.decoder = Decoder(self.hyperParams)

        indexes = []
        for idx in range(len(trainExamples)):
            indexes.append(idx)

        encoder_parameters = filter(lambda p: p.requires_grad, self.encoder.parameters())
        encoder_optimizer = torch.optim.Adam(encoder_parameters, lr = self.hyperParams.learningRate)

        decoder_parameters = filter(lambda p: p.requires_grad, self.decoder.parameters())
        decoder_optimizer = torch.optim.Adam(decoder_parameters, lr = self.hyperParams.learningRate)

        for iter in range(self.hyperParams.maxIter):
            print('###Iteration' + str(iter) + "###")
            random.shuffle(indexes)
            self.encoder.train()
            self.decoder.train()
            for updateIter in range(len(trainExamples)):
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

                encoder_hidden = self.encoder.init_hidden()
                encoder_output, encoder_hidden = self.encoder(trainExamples[updateIter].postWordIndexs, encoder_hidden)

                decoder_hidden = self.decoder.initHidden()
                response_len = trainExamples[updateIter].responseLen
                last_word = torch.autograd.Variable(torch.LongTensor([self.hyperParams.responseStartID]))
                loss = 0
                for idx in range(response_len + 1):
                    decoder_output, decoder_hidden = self.decoder(encoder_hidden, decoder_hidden, last_word)
                    loss += torch.nn.functional.nll_loss(decoder_output, trainExamples[updateIter].responseWordIndexs[0][idx])
                    wordId = self.getMaxIndex(decoder_output)
                    last_word = torch.autograd.Variable(torch.LongTensor([wordId]))
                loss.backward()
                print('Current: ', updateIter + 1, ", Cost:", loss.data[0])
                encoder_optimizer.step()
                decoder_optimizer.step()

        self.encoder.eval()
        self.decoder.eval()
        for idx in range(len(devExamples)):
            self.predict(devExamples[idx])

    def predict(self, exam):
        encoder_hidden = self.encoder.init_hidden()
        encoder_output, encoder_hidden = self.encoder(exam.postWordIndexs, encoder_hidden)
        decoder_hidden = self.decoder.initHidden()
        sent = []
        last_word = torch.autograd.Variable(torch.LongTensor([self.hyperParams.responseStartID]))
        for idx in range(self.hyperParams.maxResponseLen):
            decoder_output, decoder_hidden = self.decoder(encoder_hidden, decoder_hidden, last_word)
            wordId = self.getMaxIndex(decoder_output)
            last_word = torch.autograd.Variable(torch.LongTensor([wordId]))
            wordStr = self.hyperParams.responseWordAlpha.from_id(wordId)
            if wordStr == self.hyperParams.end:
                break
            sent.append(wordStr)
        print(sent)



    def getMaxIndex(self, decoder_output):
        max = decoder_output.data[0][0]
        maxIndex = 0
        for idx in range(1, self.hyperParams.responseWordNum):
            if decoder_output.data[0][idx] > max:
                max = decoder_output.data[0][idx]
                maxIndex = idx
        return maxIndex





parser = OptionParser()
parser.add_option("--train", dest="trainFile",
                  help="train dataset")

parser.add_option("--dev", dest="devFile",
                  help="dev dataset")

parser.add_option("--test", dest="testFile",
                  help="test dataset")


(options, args) = parser.parse_args()
l = Trainer()
l.train(options.trainFile, options.devFile, options.testFile)


