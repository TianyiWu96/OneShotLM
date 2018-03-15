import torch, torch.nn as nn, math, numpy as np, torch.nn.functional as F
from torch.autograd import Variable
import os

class CNN_Classifier(nn.Module):

    def __init__(self, num_channel=64, img_size=28):
        super(CNN_Classifier, self).__init__()
        self.c1 = nn.Conv2d(1, num_channel, 3)
        self.c2 = nn.Conv2d(num_channel, num_channel, 3)
        self.c3 = nn.Conv2d(num_channel, num_channel, 3)
        self.c4 = nn.Conv2d(num_channel, num_channel, 3)
        self.r = F.relu
        self.bn = nn.BatchNorm2d(num_channel)
        self.mp = nn.MaxPool2d(2, 2)
        nn.init.xavier_uniform(self.c1.weight)
        nn.init.xavier_uniform(self.c2.weight)
        nn.init.xavier_uniform(self.c3.weight)
        nn.init.xavier_uniform(self.c4.weight)

    def forward(self, img_input):
        x = img_input
        x = self.r(self.bn(self.c1(x)))
        x = self.r(self.bn(self.c2(x)))
        x = self.mp(x)
        x = self.r(self.bn(self.c3(x)))
        x = self.r(self.bn(self.c4(x)))
        x = self.mp(x)
        x = x.view(x.shape[0], -1)
        return x

class Linear(nn.Module):
    
    def __init__(self, lev=0,act=F.sigmoid,in_size=1024, mem_size=1024,out_size=1024,):
        super(Linear, self).__init__()
        self.w = Variable(torch.ones(in_size, out_size), requires_grad=True).cuda()
        self.u = Variable(torch.ones(mem_size, out_size), requires_grad=True).cuda()
        self.b = Variable(torch.zeros(out_size), requires_grad=True).cuda()
        nn.init.xavier_uniform(self.w.data)
        nn.init.xavier_uniform(self.u.data)
        self.act=act
        self.wl=[]
        self.bl=[]
        self.lev=lev
        for i in range(self.lev):
            self.wl.append(Variable(torch.ones(out_size, out_size), requires_grad=True).cuda())
            self.bl.append(Variable(torch.ones(out_size), requires_grad=True).cuda())
            nn.init.xavier_uniform(self.wl[i].data)
        
    def forward(self, input,_c):    
        _w = self.w.unsqueeze(0).expand(input.shape[0], -1, -1).bmm(input.unsqueeze(2)).squeeze(-1)    
        _u = self.u.unsqueeze(0).expand(input.shape[0], -1, -1).bmm(_c.unsqueeze(2)).squeeze(-1)
        _out = self.act(_w + _u + self.b)
        for i in range(self.lev):
            _w = self.wl[i].unsqueeze(0).expand(_out.shape[0], -1, -1).bmm(_out.unsqueeze(2)).squeeze(-1)    
            _out = F.relu(_w+self.bl[i])
        return _out
        
class LSTM(nn.Module):

    def __init__(self, in_size=1024, num_img=21, mem_size=1024):
        super(LSTM, self).__init__()
        self.num_img = num_img
        self.mem_size = mem_size
        self.fNN=Linear(3)
        self.iNN=Linear(3)
        self.oNN=Linear(4,F.tanh)
        self.cNN=Linear(3,F.tanh)
        self.in_c = Variable(torch.zeros( self.mem_size), requires_grad=True).cuda()
        
    def forward(self, input):
        h = []
        _c = self.in_c.unsqueeze(0).expand(input[0].shape[0],-1)
        for i in range(self.num_img):
            ft = self.fNN.forward(input[i],_c)
            it = self.iNN.forward(input[i],_c)
            _LSTM__F = self.cNN.forward(input[i],_c)
            ct = ft * _c + it * _LSTM__F
            _c= ct
            #print(ct.shape,_c.shape,ft.shape)
        for i in range(self.num_img):
            ot = self.oNN.forward(input[i],_c)
            h.append(ot)

        return h
        
class MatchingNetwork(nn.Module):

    def __init__(self):
        super(MatchingNetwork, self).__init__()
        self.cnn = CNN_Classifier().cuda()
        self.lstm = LSTM()

    def forward(self, support_set, support_set_y, target, target_y):
        x = []
        for i in range(support_set.data.shape[1]):
            x.append(self.cnn(support_set[:, i, :, :]))
        y = self.cnn(target)
        x.append(y)
        self.lstm(x)
        y = x[len(x) - 1]
        x = x[0:len(x) - 1]
        z = []
        for i in range(support_set.data.shape[1]):
            z.append(x[i].unsqueeze(1).bmm(y.unsqueeze(2)) * torch.rsqrt(x[i].unsqueeze(1).bmm(x[i].unsqueeze(2))))

        z = torch.stack(z).squeeze(-1).squeeze(-1).t()
        output = F.softmax(z, dim=1)
        values, indices = output.max(1)
        accuracy = torch.mean((indices == target_y).float())
        crossentropy_loss = F.cross_entropy(output, target_y.long())
        return (
         accuracy, crossentropy_loss)
        
class unbatched_Linear(nn.Module):
    
    def __init__(self, lev=0,act=F.sigmoid,in_size=1024, mem_size=1024,out_size=1024,):
        super(Linear, self).__init__()
        self.w = Variable(torch.ones(in_size, out_size), requires_grad=True).cuda()
        self.u = Variable(torch.ones(mem_size, out_size), requires_grad=True).cuda()
        self.b = Variable(torch.zeros(out_size), requires_grad=True).cuda()
        nn.init.xavier_uniform(self.w.data)
        nn.init.xavier_uniform(self.u.data)
        self.act=act
        self.wl=[]
        self.bl=[]
        self.lev=lev
        for i in range(self.lev):
            self.wl.append(Variable(torch.ones(out_size, out_size), requires_grad=True).cuda())
            self.bl.append(Variable(torch.ones(out_size), requires_grad=True).cuda())
            nn.init.xavier_uniform(self.wl[i].data)
        
    def forward(self, input,_c):
        _w = self.w.mm(input.unsqueeze(2)).squeeze(-1)
        _u = self.u.mm(input.unsqueeze(2)).squeeze(-1)
        _out = self.act(_w + _u + self.b)
        for i in range(self.lev):
            _w = self.wl[i].mm(_out.unsqueeze(2)).squeeze(-1)    
            _out = F.relu(_w+self.bl[i])
        return _out
        
class MyEmbedding(nn.Module)

    def __init__(self,num_word=10000,num_dim=500):
        super(MyEmbedding, self).__init__()
        self.Emb=nn.Embedding(num_word,num_dim)
        self.word2id={}
        self.word2id[''<blank_token>'']=1
        self.num_id=1
    
    def forward(self, input,use_blank=True):
        # input :  {list of word including answer and blank} 
        output=[]
        for k in range(len(input)):
            if input[k]=='<blank_token>' and use_blank:
                word=input[len(input)-1]
            else:
                word=input[k]
            if not self.word2id.has_key(word):
                self.num_id+=1
                word2id[word]=self.num_id;
            if k!=len(input)-1:
                output.append(word2id[word])
        output=torch.from_numpy(np.array(output))
        return output,self.Emb.input(output)

class EnvLSTM(nn.Module):

    def __init__(self, in_size=1024, mem_size=1024,out_size=1024):
        super(EnvLSTM, self).__init__()
        self.num_img = num_img
        self.fNNL=unbatched_Linear(0,F.sigmoid,in_size,mem_size,out_size)
        self.iNNL=unbatched_Linear(0,F.sigmoid,in_size,mem_size,out_size)
        self.oNNL=unbatched_Linear(0,F.tanh,mem_size,mem_size,out_size)
        self.cNNL=unbatched_Linear(0,F.tanh,in_size,mem_size,out_size)
        self.in_cL = Variable(torch.zeros(self.mem_size), requires_grad=True).cuda()
        self.fNNR=unbatched_Linear(0,F.sigmoid,in_size,mem_size,out_size)
        self.iNNR=unbatched_Linear(0,F.sigmoid,in_size,mem_size,out_size)
        self.oNNR=unbatched_Linear(0,F.tanh,mem_size,mem_size,out_size)
        self.cNNR=unbatched_Linear(0,F.tanh,in_size,mem_size,out_size)
        self.in_cR = Variable(torch.zeros(self.mem_size), requires_grad=True).cuda()
        
    def forward(self, input):
        #input is list : len_sentence*num_emb
        h = []
        _c = self.in_cL
        Lenv=[]
        for i in range(input.data.shape[0]):
            ft = self.fNNL.forward(input[i],_c)
            it = self.iNNL.forward(input[i],_c)
            _LSTM__F = self.cNN.forward(input[i],_c)
            Lenv.append(_c)
            _c=ft * _c + it * _LSTM__F
        Renv=[]
        _c = self.in_cR.unsqueeze(0).expand(input[0].shape[0],-1)
        for i in range(input.data.shape[1]-1,0,-1):
            ft = self.fNNR.forward(input[i],_c)
            it = self.iNNR.forward(input[i],_c)
            _LSTM__F = self.cNN.forward(input[i],_c)
            Renv.append(_c)
            _c=ft * _c + it * _LSTM__F
        
        for i in range(input.data.shape[0]):
            ot = self.oNN.forward(Lenv[i],Renv[input.data.shape[0]-i-1])
            h.append(ot)
        #output is len_sentence_*num_feat
        return h

class matching_net_nlp(nn.Module)

    def __init__(self):
        super(matching_net_nlp, self).__init__()
        self.emb = MyEmbedding(10000,500)
        self.lstm = EnvLSTM(500,1000,500)
        
    def forward_slow(self, support_set, target,use_train=True,fce=False):
        # support_set list : batch_size * sequence_size * {list of word including answer and blank} 
        # target_set list : batch_size * sequence_size * {list of word including answer and blank}
        _support_set=[]
        _support_id=[]
        for j in range(len(support_set[0])):
            _support_set.append([])
            _support_id.append([])
            for i in range(len(support_set)):
                id,emb=emb(support_set[i][j],use_blank=False)
                env=EnvLSTM(emb)
                blank_env=0
                for k in range(len(support_set[i][j])):
                    if support_set[i][j][k]=='<blank_token>':
                        blank_env=env[k]
                        blank_id=id[k]
                        break
                _support_set[j].append(blank_env)
                _support_id[j].append(blank_id)
            _support_set[j]=torch.stack(_support_set[j])
        _target=[]
        target_y=[]
        for i in range(len(target)):
            id,emb=emb(target[i],use_blank=False)
            env=EnvLSTM(emb)
            blank_env=0
            for k in range(len(target[i])):
                if target[i][k]=='<blank_token>':
                    blank_env=env[k]
                    blank_id=id[k]
                    break
            _target.append(blank_env)
            blank_ans=-1
            for j in range(len(support_set[0])):
                if _support_id[j][i]==blank_id:
                    blank_ans=j
                    break
            target_y.append(blank_ans)
        _target=torch.stack(_target)
        
        x = _support_set
        y = _target
        if fce:
            x.append(y)
            self.lstm(x)
            y = x[len(x) - 1]
            x = x[0:len(x) - 1]
        z = []
        for i in range(support_set.data.shape[1]):
            z.append(x[i].unsqueeze(1).bmm(y.unsqueeze(2)) * torch.rsqrt(x[i].unsqueeze(1).bmm(x[i].unsqueeze(2))))

        z = torch.stack(z).squeeze(-1).squeeze(-1).t()
        output = F.softmax(z, dim=1)
        values, indices = output.max(1)
        accuracy = torch.mean((indices == target_y).float())
        crossentropy_loss = F.cross_entropy(output, target_y.long())
        return (
         accuracy, crossentropy_loss)

