class HyperParams:
    def __init__(self):
        self.wordNum = 0
        self.labelSize = 0

        self.unk = '-unk-'
        self.padding = '-padding-'

        self.start = '-start-'
        self.end = '-end-'
        self.maxResponseLen = 30


        self.postWordNum = 0
        self.postUnkWordID = 0
        self.postPaddingID = 0
        self.postStartID = 0
        self.postEndID = 0

        self.responseWordNum = 0
        self.responseUnkWordID = 0
        self.responsePaddingID = 0
        self.responseStartID = 0
        self.responseEndID = 0

        self.maxIter = 10000
        self.verboseIter = 20
        self.wordCutOff = 0
        self.wordEmbSize = 100
        self.wordFineTune = True
        self.wordEmbFile = "E:\\py_workspace\\mySeq2Seq\\data\\ctb60.50d.vec"
        #self.wordEmbFile = ""
        self.dropProb = 0.1
        self.rnnHiddenSize = 50
        self.hiddenSize = 100
        self.thread = 1
        self.learningRate = 0.001
        self.maxInstance = -1
        self.batch = 12

        self.postWordAlpha = Alphabet()
        self.responseWordAlpha = Alphabet()
    def show(self):
        print('wordCutOff = ', self.wordCutOff)
        print('wordEmbSize = ', self.wordEmbSize)
        print('wordFineTune = ', self.wordFineTune)
        print('rnnHiddenSize = ', self.rnnHiddenSize)
        print('learningRate = ', self.learningRate)
        print('batch = ', self.batch)

        print('maxInstance = ', self.maxInstance)
        print('maxIter =', self.maxIter)
        print('thread = ', self.thread)
        print('verboseIter = ', self.verboseIter)


class Alphabet:
    def __init__(self):
        self.max_cap = 1e8
        self.m_size = 0
        self.m_b_fixed = False
        self.id2string = []
        self.string2id = {}

    def from_id(self, qid, defineStr = ''):
        if int(qid) < 0 or self.m_size <= qid:
            return defineStr
        else:
            return self.id2string[qid]

    def from_string(self, str):
        if str in self.string2id:
            return self.string2id[str]
        else:
            if not self.m_b_fixed:
                newid = self.m_size
                self.id2string.append(str)
                self.string2id[str] = newid
                self.m_size += 1
                if self.m_size >= self.max_cap:
                    self.m_b_fixed = True
                return newid
            else:
                return -1

    def clear(self):
        self.max_cap = 1e8
        self.m_size = 0
        self.m_b_fixed = False
        self.id2string = []
        self.string2id = {}

    def set_fixed_flag(self, bfixed):
        self.m_b_fixed = bfixed
        if (not self.m_b_fixed) and (self.m_size >= self.max_cap):
            self.m_b_fixed = True

    def initial(self, elem_state, cutoff = 0):
        for key in elem_state:
            if  elem_state[key] > cutoff:
                self.from_string(key)
        self.set_fixed_flag(True)
