# utils.py

import torch
import torchtext
from torchtext.legacy import data
from torchtext.vocab import Vectors
import spacy
import pandas as pd
import numpy as np
# from sklearn.metrics import accuracy_score
import en_core_web_sm

import pickle

class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.train_iterator = None
        self.test_iterator = None
        self.val_iterator = None
        self.vocab = []
        self.word_embeddings = {}
    
    def parse_label(self, label):
        '''
        Get the actual labels from label string
        Input:
            label (string) : labels of the form '__label__2'
        Returns:
            label (int) : integer value corresponding to label string
        '''
        return int(label.strip()[-1])

    def get_pandas_df(self, filename):
        '''
        Load the data into Pandas.DataFrame object
        This will be used to convert data to torchtext object
        '''
        with open(filename, 'r') as datafile:     
            data = [line.strip().split(',', maxsplit=1) for line in datafile]
            data_text = list(map(lambda x: x[1], data))
            data_label = list(map(lambda x: self.parse_label(x[0]), data))

        full_df = pd.DataFrame({"text":data_text, "label":data_label})
        return full_df
    
    def load_data(self, w2v_file, train_file, test_file, val_file=None):
        '''
        Loads the data from files
        Sets up iterators for training, validation and test data
        Also create vocabulary and word embeddings based on the data
        
        Inputs:
            w2v_file (String): absolute path to file containing word embeddings (GloVe/Word2Vec)
            train_file (String): absolute path to training file
            test_file (String): absolute path to test file
            val_file (String): absolute path to validation file
        '''

        NLP = en_core_web_sm.load()
        tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]
        # sent_test = "hello world!"
        # print(f"Tokenized sentence: {tokenizer(sent_test)}") # ['hello', 'world', '!']
        
        # Creating Field for data
        TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)
        LABEL = data.Field(sequential=False, use_vocab=False)
        datafields = [("text",TEXT),("label",LABEL)]
        
        # Load data from pd.DataFrame into torchtext.data.Dataset
        train_df = self.get_pandas_df(train_file)
        train_examples = [data.Example.fromlist(i, datafields) for i in train_df.values.tolist()]
        # print(f"train_examples: {train_examples}") # list of torchtext.legacy.data.example.Example
        # dump train_examples to file
        with open('train_examples.pkl', 'wb') as f:
            pickle.dump(train_examples, f)
        train_data = data.Dataset(train_examples, datafields)
        train_val_data = train_data
        
        test_df = self.get_pandas_df(test_file)
        test_examples = [data.Example.fromlist(i, datafields) for i in test_df.values.tolist()]
        # dump test_examples to file
        with open('test_examples.pkl', 'wb') as f:
            pickle.dump(test_examples, f)
        test_data = data.Dataset(test_examples, datafields)
        
        # If validation file exists, load it. Otherwise get validation data from training data
        if val_file:
            val_df = self.get_pandas_df(val_file)
            val_examples = [data.Example.fromlist(i, datafields) for i in val_df.values.tolist()]
            val_data = data.Dataset(val_examples, datafields)
        else:
            import random
            random.seed(100) # make it deterministic
            train_data, val_data = train_data.split(split_ratio=0.8)
        
        TEXT.build_vocab(train_data, vectors=Vectors(w2v_file))
        self.word_embeddings = TEXT.vocab.vectors
        self.vocab = TEXT.vocab

        # dump train_data, val_data, test_data vocab, word_embeddings to pickle
        # with open('train_data.pkl', 'wb') as f:
        #     pickle.dump((train_data), f)
        # with open('val_data.pkl', 'wb') as f:
        #     pickle.dump((val_data), f)
        # with open('test_data.pkl', 'wb') as f:
        #     pickle.dump((test_data), f)
        with open('vocab.pkl', 'wb') as f:
            pickle.dump((self.vocab), f)
        with open('word_embeddings.pkl', 'wb') as f:
            pickle.dump((self.word_embeddings), f)

        self.train_iterator = data.BucketIterator(
            (train_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=True)
        
        self.val_iterator, self.test_iterator = data.BucketIterator.splits(
            (val_data, test_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=False)
        
        self.train_val_iterator = data.BucketIterator(
            (train_val_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=True)

        print ("Loaded {} training examples".format(len(train_data)))
        print ("Loaded {} test examples".format(len(test_data)))
        print ("Loaded {} validation examples".format(len(val_data)))

def load_data_from_pickle(config, w2v_file='../data/glove.840B.300d.txt'):
    NLP = en_core_web_sm.load()
    tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]
    TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)
    LABEL = data.Field(sequential=False, use_vocab=False)
    datafields = [("text",TEXT),("label",LABEL)]

    with open('train_examples.pkl', 'rb') as f:
        train_examples = pickle.load(f)
    train_data = data.Dataset(train_examples, datafields)
    with open('test_examples.pkl', 'rb') as f:
        test_examples = pickle.load(f)
    test_data = data.Dataset(test_examples, datafields)
    import random
    random.seed(100) # make it deterministic
    train_val_data = train_data
    train_data, val_data = train_data.split(split_ratio=0.8)

    TEXT.build_vocab(train_data, vectors=Vectors(w2v_file))

    train_iterator = data.BucketIterator(
        (train_data),
        batch_size=config.batch_size,
        sort_key=lambda x: len(x.text),
        repeat=False,
        shuffle=True)

    val_iterator, test_iterator = data.BucketIterator.splits(
        (val_data, test_data),
        batch_size=config.batch_size,
        sort_key=lambda x: len(x.text),
        repeat=False,
        shuffle=False)

    train_val_iterator = data.BucketIterator(
        (train_val_data),
        batch_size=config.batch_size,
        sort_key=lambda x: len(x.text),
        repeat=False,
        shuffle=True)

    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    with open('word_embeddings.pkl', 'rb') as f:
        word_embeddings = pickle.load(f)

    return train_iterator, val_iterator, test_iterator, vocab, word_embeddings, train_val_iterator



def evaluate_model(model, iterator):
    all_preds = []
    all_y = []
    for idx,batch in enumerate(iterator):
        # if torch.cuda.is_available():
        #     x = batch.text.cuda()
        # else:
        x = batch.text
        y_pred = model(x)
        predicted = torch.max(y_pred.cpu().data, 1)[1] + 1
        all_preds.extend(predicted.numpy())
        all_y.extend(batch.label.numpy())
    # score = accuracy_score(all_y, np.array(all_preds).flatten())
    # return score
    return np.mean(np.array(all_preds) == np.array(all_y))

def save_dataset(iterator, filename='dataset.pkl'):
    datalist = []
    for idx, batch in enumerate(iterator):
        x = batch.text.numpy()
        # reshape x from (size, 1) to (size,)
        x = x.reshape(-1)
        y = batch.label.numpy()
        y[0] = y[0] - 1  # Adjust label to be zero-indexed
        # if x size < 94, pad with 1, otherwise truncate to 94
        if x.shape[0] < 94:
            x = np.pad(x, (0, 94 - x.shape[0]), 'constant', constant_values=1)
        else:
            x = x[:94]
        if idx == 0:
            print(f"save_dataset: x={x}, x.shape={x.shape}, x.dtype={x.dtype}")
            print(f"save_dataset: y={y}, y.shape={y.shape}, y.dtype={y.dtype}")
        datalist.append((x, y))
    # save datalist to file
    with open(filename, 'wb') as f:
        pickle.dump(datalist, f)

def convert_ext_dataset(w2v_file='../data/glove.840B.300d.txt'):
    NLP = en_core_web_sm.load()
    tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]

    TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)
    LABEL = data.Field(sequential=False, use_vocab=False)
    datafields = [("text",TEXT),("label",LABEL)]

    with open('train_examples.pkl', 'rb') as f:
        train_examples = pickle.load(f)
    train_data = data.Dataset(train_examples, datafields)
    # with open('test_examples.pkl', 'rb') as f:
    #     test_examples = pickle.load(f)
    # test_data = data.Dataset(test_examples, datafields)

    import random
    random.seed(100) # make it deterministic
    train_val_data = train_data
    train_data, val_data = train_data.split(split_ratio=0.8)

    TEXT.build_vocab(train_data, vectors=Vectors(w2v_file))

    with open('dbpedia14_train_datasetlist_text.pkl', 'rb') as f:
        ext_data_text_list = pickle.load(f)

    datalist = []
    for i, text in enumerate(ext_data_text_list):
        prompt = text.strip().lower()
        tokens = tokenizer(prompt)
        ids = [TEXT.vocab.stoi.get(t, TEXT.vocab.stoi[TEXT.unk_token]) for t in tokens]
        # torch_ids = torch.LongTensor(ids).unsqueeze(1)
        torch_ids = torch.tensor(ids, dtype=torch.int64)
        if torch_ids.shape[0] < 94:
            torch_ids = torch.nn.functional.pad(torch_ids, (0, 94 - torch_ids.shape[0]), value=1)
        else:
            torch_ids = torch_ids[:94]
        datalist.append((torch_ids.numpy()))

        if i == 0:
            print(f"prompt: {prompt}")
            print(f"tokens: {tokens}")
            print(f"ids: {ids}")
            print(f"torch_ids: {torch_ids}, shape={torch_ids.shape}, dtype={torch_ids.dtype}")

    with open('dbpedia14_train_datasetlist.pkl', 'wb') as f:
        pickle.dump(datalist, f)