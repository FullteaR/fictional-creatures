import Levenshtein
import random

class MarkovMonsterNameGenerator:
    
    def __init__(self, n=2):
        self.n = n  # Nグラムのサイズ
        self.transitions = {}
        self.bos = "^"
        self.eos = "$"
        self.originals = []
        self.max_length = 7
        self.min_length = 3

    def train(self, names):
        self.originals += names
        for name in names:
            # 文頭・文末マークを追加
            name = self.bos * (self.n) + name + self.eos
            for i in range(len(name) - self.n):
                gram = name[i:i+self.n]
                next_char = name[i+self.n]
                if gram not in self.transitions:
                    self.transitions[gram] = []
                self.transitions[gram].append(next_char)

    def valid_name(self, name):
        if len(name) > self.max_length or len(name)< self.min_length:
            return False
        for original in self.originals:
            if name in original:
                return False
            if len(original)>3 and original in name:
                return False
            distance = Levenshtein.distance(name, original)
            if distance <= 2 and len(name)>5:
                return False
            if distance <= 1:
                return False
        return True

    def generate(self):
        for _ in range(100):
            name_cand = self.__generate()
            if self.valid_name(name_cand):
                break
        return name_cand
            

    def __generate(self):
        gram = self.bos * (self.n)
        result = ''
        random.seed()
        while True:
            next_chars = self.transitions.get(gram)
            if not next_chars:
                break

            next_char = random.choice(next_chars)
            if next_char == self.eos  or len(result) >= self.max_length + 1 :
                break
            result += next_char
            gram = (gram + next_char)[-self.n:]
        return result

    def train_from_file(self, filepath):
        with open(filepath, encoding='utf-8') as f:
            names = [line.strip() for line in f if line.strip()]
        self.train(names)