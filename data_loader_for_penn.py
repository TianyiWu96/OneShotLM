import nltk
from collections import Counter
from nltk.corpus import treebank
import numpy as np
from random import shuffle
from random import randint
from copy import deepcopy

class PTB():
    def __init__(self):
        nltk.download('treebank')
        ids = nltk.corpus.treebank.fileids()
        self.sents=[]
        for id in ids:
            self.sents+=list(treebank.sents(id))
        self.wc={}
        for i in range(len(self.sents)):
            for j in range(len(self.sents[i])):
                if not self.sents[i][j] in self.wc:
                    self.wc[self.sents[i][j]]=[]
                self.wc[self.sents[i][j]].append((i,j))
        self.word_set=[]
        frequency = 10
        for i in self.wc:
            if len(self.wc[i])>= frequency and i.isalpha():
                self.word_set.append(i)
        self.n=len(self.word_set)
        self.n_s=len(self.word_set)
        print(len(self.sents))
        print(len(self.wc))
        print(len(self.word_set))
        
    def get_question(self,num_ways,is_train):
        while True:
            if self.count==self.n:
                self.count=0
                shuffle(self.ask)
            _w=self.word_set[self.ask[self.count]]
            pos=[]
            for i in range(len(self.wc[_w])):
                if (self.wc[_w][i][0]<self.train_s and is_train) or (self.wc[_w][i][0]>=self.train_s and not is_train):
                    pos.append(self.wc[_w][i])
            self.count+=1
            if(len(pos)>=2):
                break

        shuffle(pos)
        _support={}
        _support[self.sents[pos[0][0]][pos[0][1]]]=pos[0]
        _target=pos[1]
        for j in range(1,num_ways):
            while True:
                k=randint(0,self.train_s-1)
                if not self.word_set[k] in _support:
                    _w=self.word_set[k]
                    pos=[]
                    for i in range(len(self.wc[_w])):
                        if (self.wc[_w][i][0]<self.train_s and is_train) or (self.wc[_w][i][0]>=self.train_s and not is_train):
                            pos.append(self.wc[_w][i]) 
                    if(len(pos)>=2):
                        break
            _i=pos[randint(0,len(pos)-1)]
            _support[self.sents[_i[0]][_i[1]]]=_i
            
        set_support=list(_support)
        shuffle(set_support)
        support=[]
        for i in range(len(set_support)):
            support.append([])        
            id,pos=_support[set_support[i]]
            support[i]=deepcopy(self.sents[id])
            support[i].append(support[i][pos])
            support[i][pos]="<blank_token>"
        id,pos=_target
        target=deepcopy(self.sents[id])
        target.append(target[pos])
        target[pos]="<blank_token>"         
        return support,target

    def build_experiment(self):
        self.count=0
        self.ask=list(range(self.n))
        shuffle(self.ask)
        self.count_s=0
        self.r_sent=list(range(self.n_s))
        shuffle(self.r_sent)
        self._r_sent=list(range(self.n_s))
        for i in range(len(self.r_sent)):
            self._r_sent[self.r_sent[i]]=i
        
        self.train_s=int(self.n_s*0.7)
        
    def get_batch(self,num_batch,num_ways,is_train):
        support=[]
        target=[]
        for i in range(num_batch):
            _s,_t=self.get_question(num_ways,is_train)
            support.append(_s)
            target.append(_t)
        return support,target
        
    def get_batch_sentences(self,num_batch):
        support=[]
        for i in range(num_batch):
            while True:
                if self.count_s==self.train_s:
                    self.count_s=0
                    shuffle(self.r_sent[0:self.train_s])
                    for j in range(len(self.r_sent)):
                        self._r_sent[self.r_sent[i]]=i
                if len(self.sents[self.r_sent[self.count_s]])>1:
                    break
                self.count_s+=1               
            support.append(deepcopy(self.sents[self.r_sent[self.count_s]]))
            self.count_s+=1
        return support