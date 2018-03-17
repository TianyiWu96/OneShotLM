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
        words = list(treebank.words())
        frequency = 10
        self.word_set = []
        for word, count in Counter(words).items():
            if count >= frequency and word.isalpha():
                self.word_set.append(word)
        ids = nltk.corpus.treebank.fileids()
        self.n=len(self.word_set)
        self.word2Id = {}
        for i in self.word_set:
            self.word2Id[i]=[]
        self.sents=[]
        self.wc={}
        for id in ids:
            self.sents+=list(treebank.sents(id))
        for i in range(len(self.sents)):
            for j in range(len(self.sents[i])):
                if self.sents[i][j] in self.word_set:
                    self.word2Id[self.sents[i][j]].append((i,j))
                self.wc[self.sents[i][j]]=1
        print(len(self.wc))
        self.n=len(self.word_set)
        
    def get_question(self,Input,num_ways,range_l,range_r):
        _support={}
        _support[Input]=randint(range_l,range_r)
        for j in range(1,num_ways):
            while True:
                k=randint(0,self.n-1)
                if not k in _support:
                    _support[k]=randint(range_l,range_r)
                    break
        while True:
            k=randint(range_l,range_r)
            if k!=_support[Input]:
                _target=k
                break
        set_support=list(_support)
        shuffle(set_support)
        support=[]
        for i in range(len(set_support)):
            support.append([])        
            id,pos=self.word2Id[self.word_set[set_support[i]]][_support[set_support[i]]]
            support[i]=deepcopy(self.sents[id])
            support[i].append(support[i][pos])
            support[i][pos]="<blank_token>"
            #print(support[i])
        id,pos=self.word2Id[self.word_set[Input]][_target]
        target=deepcopy(self.sents[id])
        target.append(target[pos])
        target[pos]="<blank_token>"         
        return support,target
    
    '''
    def get_question(self,Input,num_ways,range_l,range_r):
        _support={}
        _support[Input]=0
        for j in range(1,num_ways):
            k=(Input+j)%self.n
            _support[k]=0#randint(range_l,range_r)
        _target=1
        set_support=list(_support)
        shuffle(set_support)
        support=[]
        for i in range(len(set_support)):
            support.append([])        
            id,pos=self.word2Id[self.word_set[set_support[i]]][_support[set_support[i]]]
            support[i]=deepcopy(self.sents[id])
            support[i].append(support[i][pos])
            support[i][pos]="<blank_token>"
            #print(support[i])
        id,pos=self.word2Id[self.word_set[Input]][_target]
        target=deepcopy(self.sents[id])
        target.append(target[pos])
        target[pos]="<blank_token>"         
        return support,target
    '''    
    def build_experiment(self):
        #self.n=20
        for i in range(self.n):
            shuffle(self.word2Id[self.word_set[i]])
        self.count=0
        self.ask=list(range(self.n))
        shuffle(self.ask)
        self.count_s=0
        self.r_sent=[]
        for i in range(self.n):
            for j in range(0,7):
                self.r_sent.append(self.word2Id[self.word_set[i]][j][0])    
        self.r_sent=list(set(self.r_sent))
        shuffle(self.r_sent)
        self.n_s=len(self.r_sent)
        print(self.n_s)
        
    def get_batch(self,num_batch,num_ways,is_train):
        if is_train:
            range_l=0
            range_r=6
        else:
            range_l=7
            range_r=9
        support=[]
        target=[]
        for i in range(num_batch):
            if self.count==self.n:
                self.count=0
                shuffle(self.ask)
            _s,_t=self.get_question(self.ask[self.count],num_ways,range_l,range_r)
            support.append(_s)
            target.append(_t)
            self.count+=1
        return support,target
        
    def get_batch_sentences(self,num_batch):
        support=[]
        for i in range(num_batch):
            if self.count_s==self.n_s:
                self.count_s=0
                shuffle(self.r_sent)
            support.append(deepcopy(self.sents[self.r_sent[self.count_s]]))
            self.count_s+=1
        return support