from data_loader_for_penn import PTB
from mymatchingnet import matching_net_nlp
import torch.backends.cudnn as cudnn
import torch
# Experiment setup
batch_size = 20
fce = True
classes_per_set = 5
samples_per_class = 1

# Training setup
total_epochs = 100
total_train_batches = 1000
total_test_batches = 100


matchNet=matching_net_nlp(classes_per_set)
cudnn.benchmark = True  # set True to speedup
torch.cuda.manual_seed_all(2017)

optimizer = torch.optim.Adam(matchNet.parameters())

data=PTB()
data.build_experiment()
def RunBatch(total_batches,is_train):
    total_c_loss = 0.0
    total_accuracy = 0.0
    total_batches=int(total_batches/batch_size)
    for i in range(total_batches):
        support_set, target= data.get_batch(batch_size,classes_per_set,is_train)    
        acc, c_loss = matchNet(support_set,target)
        if is_train:
            optimizer.zero_grad()
            c_loss.backward()
            optimizer.step() 
        total_c_loss += c_loss.data[0]
        total_accuracy += acc.data[0]     
    total_c_loss = total_c_loss / total_train_batches
    total_accuracy = total_accuracy / total_train_batches
    return total_c_loss,total_accuracy

for e in range(total_epochs):
    total_c_loss,total_accuracy=RunBatch(total_train_batches,True)
    print("Train: Epoch ",e,": loss=",total_c_loss," acc=",total_accuracy)
    total_c_loss,total_accuracy=RunBatch(total_test_batches,False)
    print("Test: Epoch ",e,": loss=",total_c_loss," acc=",total_accuracy) 