# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: BoyuanJiang
# College of Information Science & Electronic Engineering,ZheJiang University
# Email: ginger188@gmail.com
# Copyright (c) 2017

# @Time    :17-8-27 10:46
# @FILE    :data_loader.py
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
class TreeBankDataset():
    def __init__(self, batch_size, classes_per_set=20, samples_per_class=1, seed=2017, shuffle=True, use_cache=True):
        np.random.seed(seed)
        self.x = np.load('dataset.npy')
        self.x = self._process_vocabs(self.x)
        self.x = np.reshape(self.x, newshape=(-1,10)) 
        if shuffle:
            np.random.shuffle(self.x)
        self.x_train, self.x_val, self.x_test = self.x[:900], self.x[900:1000], self.x[1000:]
        print(self.x_train.shape)
        self.batch_size = batch_size # batch = 20
        self.n_classes = self.x.shape[0] # total classes
        self.classes_per_set = classes_per_set # 20 way
        self.samples_per_class = samples_per_class # 1 shot
        self.indexes = {"train": 0, "val": 0, "test": 0}
        self.datatset = {"train": self.x_train, "val": self.x_val, "test": self.x_test}
        self.use_cache = use_cache
        print(self.x_train.shape)
        if self.use_cache:
            self.cached_dataset = {"train": self.load_data_cache(self.x_train),
                                    "val": self.load_data_cache(self.x_val),
                                    "test": self.load_data_cache(self.x_test)}


    def _process_vocabs(self, data): 
        """
        process all the words and return the idx of each word in the sentence
        """
        all_words = []
        for i in data:
            all_words = all_words + i
        #list of all words in the dataset    
        vocab = sorted(list(set(all_words)))
        print("Vocab processed, {} words in total".format(len(vocab)))
        self.idx_to_word = dict(zip(range(len(vocab)), vocab))
        self.word_to_idx = dict(zip(vocab, range(len(vocab))))
        return np.array([[self.word_to_idx[word] for word in sequence] for sequence in data]) # this will count as all the 
     
   
        
    def _get_batch(self, dataset_name):
        """
        Get next batch from the dataset with name.
        :param dataset_name: The name of dataset(one of "train","val","test")
        :param augment: if rotate the images
        :return: a batch images
        """
        if self.use_cache:
            support_set_x, support_set_y, target_x, target_y = self._get_batch_from_cache(dataset_name)
        else:
            print("use_cache = false not yet implemented")
            return

        support_set_x = support_set_x.reshape((support_set_x.shape[0], support_set_x.shape[1] * support_set_x.shape[2], self.maxlen))
        support_set_y = support_set_y.reshape(support_set_y.shape[0], support_set_y.shape[1] * support_set_y.shape[2])
        return support_set_x, support_set_y, target_x, target_y

    def get_train_batch(self):
        return self._get_batch("train")

    def get_val_batch(self):
        return self._get_batch("val")

    def get_test_batch(self):
        return self._get_batch("test")

    def load_data_cache(self, data_pack):
        """ 
        load the data with batch, 
        """
        cached_dataset = []
        self.classes_idx = np.arange(data_pack.shape[0]) 
        self.samples_idx = np.arange(data_pack.shape[1]) 
        for _ in range(1000): 
            support_set_x = []
            support_set_y = [] 
            target_x = [] 
            target_y = [] 
            for i in range(self.batch_size):  
                choose_classes = np.random.choice(self.classes_idx, size= self.classes_per_set, replace=False) # choose the 20 classes
                choose_label = np.random.choice(self.classes_per_set, size=1) # choose which to query
                choose_samples = np.random.choice(self.samples_idx, size=self.samples_per_class + 1, replace=False) 
                x_temp = data_pack[choose_classes,:] # select 20 classes as support 
                x_temp = x_temp[:,choose_samples] # 
                y_temp = np.arange(self.classes_per_set)
                xx = []
                #print(x_temp[-1])
                for x in x_temp:
                    xx.append(list(x[:-1]))
                support_set_x.append(xx)
                support_set_y.append(list(y_temp)) 
                target_x.append(list(x_temp[choose_label][-1])) 
                target_y.append(y_temp[choose_label]) 
            cached_dataset.append([support_set_x, support_set_y, target_x, target_y])
        return cached_dataset 

    def _get_batch_from_cache(self, dataset_name):
        """
        
        """ 
        if self.indexes[dataset_name] >= len(self.cached_datatset[dataset_name]):
            self.indexes[dataset_name] = 0
            self.cached_datatset[dataset_name] = self.load_data_cache(self.datatset[dataset_name])
        next_batch = self.cached_datatset[dataset_name][self.indexes[dataset_name]] # load the next batch based on the last index
        self.indexes[dataset_name] += 1
        x_support_set, y_support_set, x_target, y_target = next_batch
        return x_support_set, y_support_set, x_target, y_target
