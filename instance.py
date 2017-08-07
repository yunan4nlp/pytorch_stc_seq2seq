class Instance:
    def __init__(self):
        self.post = []
        self.response = []

    def show(self):
        print(self.post)
        print(self.response)


class Example:
    def __init__(self):
        self.postWordIndexs = []
        self.postLen = 0

        self.responseWordIndexs = []
        self.responseLen = 0

    def show(self):
        print(self.postWordIndexs)
        print(self.postLen)
        print(self.responseWordIndexs)
        print(self.responseLen)

