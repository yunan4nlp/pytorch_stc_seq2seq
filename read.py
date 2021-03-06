from instance import  Instance
import torch
import re
import torch.nn as nn

class Reader:
    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def readInstances(self, path, maxInst = -1):
        insts = []
        r = open(path, encoding='utf8')
        info = []
        for line in r.readlines():
            line = line.strip()
            if line == '':
                for idx in range(1, len(info)):
                    inst = Instance()
                    post = self.clean_str(info[0])
                    inst.post = post.split(" ")
                    response = self.clean_str(info[idx])
                    inst.response = response.split(" ")
                    if maxInst == -1 or (len(insts) < maxInst):
                        insts.append(inst)
                info = []
            else:
                info.append(line)
        if len(info) != 0:
            for idx in range(1, len(info)):
                inst = Instance()
                inst.post = info[0].split(" ")
                inst.response = info[idx].split(" ")
                if maxInst == -1 or (len(insts) < maxInst):
                    insts.append(inst)
        r.close()
        return insts

    def load_pretrain(self, file, alpha, unk):
        f = open(file, encoding='utf-8')
        allLines = f.readlines()
        indexs = []
        info = allLines[0].strip().split(' ')
        embDim = len(info) - 1
        emb = nn.Embedding(alpha.m_size, embDim)
        oov_emb = torch.zeros(1, embDim).type(torch.FloatTensor)
        for line in allLines:
            info = line.strip().split(' ')
            wordID = alpha.from_string(info[0])
            if wordID >= 0:
                indexs.append(wordID)
                for idx in range(embDim):
                    val = float(info[idx + 1])
                    emb.weight.data[wordID][idx] = val
                    oov_emb[0][idx] += val
        f.close()
        count = len(indexs)
        for idx in range(embDim):
            oov_emb[0][idx] /= count
        unkID = alpha.from_string(unk)
        print('UNK ID: ', unkID)
        if unkID != -1:
            for idx in range(embDim):
                emb.weight.data[unkID][idx] = oov_emb[0][idx]
        print("Load Embedding file: ", file, ", size: ", embDim)
        oov = 0
        for idx in range(alpha.m_size):
            if idx not in indexs:
                oov += 1
        print("OOV Num: ", oov, "Total Num: ", alpha.m_size,
              "OOV Ratio: ", oov / alpha.m_size)
        print("OOV ", unk, "use avg value initialize")
        return emb, embDim

