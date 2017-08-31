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

    def getBatchFeatLabel(self, exams):
        maxPostLen = -1
        maxResponseLen = -1
        batch = self.hyperParams.batch
        for e in exams:
            if maxPostLen < e.postWordIndexs.size()[1]:
                maxPostLen = e.postWordIndexs.size()[1]
            if maxResponseLen < e.responseWordIndexs.size()[1]:
                maxResponseLen = e.responseWordIndexs.size()[1]
        postBatchFeat = torch.autograd.Variable(torch.LongTensor(batch, maxPostLen))
        responseBatchFeat = torch.autograd.Variable(torch.LongTensor(batch, maxResponseLen))

        for idx in range(batch):
            e = exams[idx]
            for idy in range(maxPostLen):
                if idy < e.postWordIndexs.size()[1]:
                    postBatchFeat.data[idx][idy] = e.postWordIndexs.data[0][idy]
                else:
                    postBatchFeat.data[idx][idy] = self.hyperParams.postPaddingID

            for idy in range(maxResponseLen):
                if idy < e.responseWordIndexs.size()[1]:
                    responseBatchFeat.data[idx][idy] = e.responseWordIndexs.data[0][idy]
                else:
                    responseBatchFeat.data[idx][idy] = self.hyperParams.responsePaddingID
        return postBatchFeat, responseBatchFeat


    def train(self, train_file, dev_file, test_file, model_file):
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

        batchBlock = len(trainExamples) // self.hyperParams.batch
        for iter in range(self.hyperParams.maxIter):
            print('###Iteration' + str(iter) + "###")
            random.shuffle(indexes)
            self.encoder.train()
            self.decoder.train()
            for updateIter in range(batchBlock):
                exams = []
                start_pos = updateIter * self.hyperParams.batch
                end_pos = (updateIter + 1) * self.hyperParams.batch
                for idx in range(start_pos, end_pos):
                    exams.append(trainExamples[indexes[idx]])
                postFeats, responseFeats = self.getBatchFeatLabel(exams)
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

                encoder_hidden = self.encoder.init_hidden(self.hyperParams.batch)
                encoder_output, encoder_hidden = self.encoder(postFeats, encoder_hidden)
                decoder_hidden = self.decoder.initHidden(self.hyperParams.batch)
                response_len =  responseFeats.size()[1]
                last_word = torch.autograd.Variable(torch.LongTensor(1, self.hyperParams.batch))
                for idx in range(self.hyperParams.batch):
                    last_word.data[0][idx] = self.hyperParams.responseStartID
                loss = 0
                for idx in range(response_len):
                    decoder_output, decoder_hidden = self.decoder(encoder_hidden, decoder_hidden, last_word)
                    loss += torch.nn.functional.nll_loss(decoder_output, responseFeats.permute(1, 0)[idx])
                    for idy in range(self.hyperParams.batch):
                        last_word.data[0][idy] = self.getMaxIndex(decoder_output[idy])
                loss.backward()
                print('Current: ', updateIter + 1, ", Cost:", loss.data[0])
                encoder_optimizer.step()
                decoder_optimizer.step()
        self.encoder.eval()
        self.decoder.eval()

        print("Save model .....")
        self.saveModel(model_file)
        print("Model model ok")

    def saveModel(self, model_file):
        torch.save([self.encoder, self.decoder], model_file)
        self.hyperParams.postWordAlpha.write(model_file + ".post")
        self.hyperParams.responseWordAlpha.write(model_file + ".response")

    def loadModel(self, model_file):
        self.encoder, self.decoder = torch.load(model_file)
        self.encoder.eval()
        self.decoder.eval()
        self.hyperParams.postWordAlpha.read(model_file + ".post")
        self.hyperParams.responseWordAlpha.read(model_file + ".response")

        self.hyperParams.postStartID = self.hyperParams.postWordAlpha.from_string(self.hyperParams.start)
        self.hyperParams.postEndID = self.hyperParams.postWordAlpha.from_string(self.hyperParams.end)
        self.hyperParams.postUnkWordID = self.hyperParams.postWordAlpha.from_string(self.hyperParams.unk)
        self.hyperParams.postPaddingID = self.hyperParams.postWordAlpha.from_string(self.hyperParams.padding)
        self.hyperParams.postWordNum = self.hyperParams.postWordAlpha.m_size

        self.hyperParams.responseStartID = self.hyperParams.responseWordAlpha.from_string(self.hyperParams.start)
        self.hyperParams.responseEndID = self.hyperParams.responseWordAlpha.from_string(self.hyperParams.end)
        self.hyperParams.responseUnkWordID = self.hyperParams.responseWordAlpha.from_string(self.hyperParams.unk)
        self.hyperParams.responsePaddingID = self.hyperParams.responseWordAlpha.from_string(self.hyperParams.padding)
        self.hyperParams.responseWordNum = self.hyperParams.responseWordAlpha.m_size


    def test(self, test_file, model_file):
        self.loadModel(model_file)
        reader = Reader()
        testInsts = reader.readInstances(test_file, self.hyperParams.maxInstance)
        testExamples = self.instance2Example(testInsts)
        for idx in range(len(testExamples)):
            self.predict(testExamples[idx])

    def predict(self, exam):
        encoder_hidden = self.encoder.init_hidden()
        encoder_output, encoder_hidden = self.encoder(exam.postWordIndexs, encoder_hidden)
        decoder_hidden = self.decoder.initHidden()
        sent = []
        last_word = torch.autograd.Variable(torch.LongTensor([self.hyperParams.responseStartID]))
        for idx in range(self.hyperParams.maxResponseLen):
            decoder_output, decoder_hidden = self.decoder(encoder_hidden, decoder_hidden, last_word)
            wordId = self.getMaxIndex(decoder_output[0])
            last_word = torch.autograd.Variable(torch.LongTensor([wordId]))
            wordStr = self.hyperParams.responseWordAlpha.from_id(wordId)
            if wordStr == self.hyperParams.end:
                break
            sent.append(wordStr)
        print(sent)



    def getMaxIndex(self, decoder_output):
        max = decoder_output.data[0]
        maxIndex = 0
        for idx in range(1, self.hyperParams.responseWordNum):
            if decoder_output.data[idx] > max:
                max = decoder_output.data[idx]
                maxIndex = idx
        return maxIndex





parser = OptionParser()
parser.add_option("--train", dest="trainFile",
                  help="train dataset")

parser.add_option("--dev", dest="devFile",
                  help="dev dataset")

parser.add_option("--test", dest="testFile",
                  help="test dataset")

parser.add_option("--model", dest="modelFile",
                  help="model file")
parser.add_option(
    "-l", "--learn", dest="learn", help="learn or test", action="store_false", default=True)

random.seed(0)
torch.manual_seed(0)
(options, args) = parser.parse_args()
l = Trainer()
if options.learn:
    l.train(options.trainFile, options.devFile, options.testFile, options.modelFile)
else:
    l.test(options.testFile,options.modelFile)


