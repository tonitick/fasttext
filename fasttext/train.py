# train.py

from utils import *
from model import *
from config import Config
import numpy as np
import sys
import torch.optim as optim
from torch import nn
import torch

import os
import pickle

if __name__=='__main__':
    # if model.pth exist, just load the model and save to onnx
    # to onnx
    # if os.path.exists('model.pth') and os.path.exists('dataset.pkl'):
    if os.path.exists('model.pth') and os.path.exists('train_examples.pkl') and os.path.exists('test_examples.pkl') and os.path.exists('vocab.pkl') and os.path.exists('word_embeddings.pkl'):
        config = Config()
        train_iterator, val_iterator, test_iterator, vocab, word_embeddings, train_val_iterator = load_data_from_pickle(config)

        # with open('dataset.pkl', 'rb') as f:
        #     vocab, word_embeddings = pickle.load(f)
        with open('x_y.pkl', 'rb') as f:
            x, y = pickle.load(f)

        print("x={}, x.shape={}".format(x, x.shape))
        print("y={}, y.shape={}".format(y, y.shape))


        model = fastText(config, len(vocab), word_embeddings)
        model.load_state_dict(torch.load('model.pth'))
        model.eval()

        y_pred = model(x)
        print('y_pred: {}'.format(y_pred))

        train_acc = evaluate_model(model, train_iterator)
        val_acc = evaluate_model(model, val_iterator)
        test_acc = evaluate_model(model, test_iterator)
        train_ori_acc = evaluate_model(model, train_val_iterator)

        print ('Final Training Accuracy (trained): {:.4f}'.format(train_acc))
        print ('Final Validation Accuracy (trained): {:.4f}'.format(val_acc))
        print ('Final Training + Validation Accuracy (trained): {:.4f}'.format(train_ori_acc))
        print ('Final Test Accuracy (trained): {:.4f}'.format(test_acc))

        # export to onnx
        sequence_length = 94  # Static sequence length
        batch_size = 1  # Static batch size
        input_ids = torch.ones((sequence_length, batch_size), dtype=torch.int64)
        model.eval()  # Ensure the model is in evaluation mode

        # save x bin input
        # take the first column of x as sample
        x_sample = x.numpy()[:, 0][0: sequence_length]
        print('x_sample: {}'.format(x_sample))
        with open('x_sample.bin', 'wb') as f:
            x_sample.tofile(f)

        torch.onnx.export(
            model,
            args=(input_ids),  # Inputs to the model
            f="fasttext.onnx",
            input_names=["input_ids"],  # Names of the input nodes
            output_names=["output"],  # Name of the output node
            dynamic_axes=None,  # No dynamic axes, implying static batch size and sequence length
            opset_version=7,  # Use opset version 14
            do_constant_folding=True,  # Whether to perform constant folding for optimization
        )

        print('model.pth exists, model exported to fasttext.onnx')
        exit(0)


    config = Config()
    train_file = '../data/ag_news.train'
    if len(sys.argv) > 2:
        train_file = sys.argv[1]
    test_file = '../data/ag_news.test'
    if len(sys.argv) > 3:
        test_file = sys.argv[2]
    
    w2v_file = '../data/glove.840B.300d.txt'
    
    dataset = Dataset(config)
    dataset.load_data(w2v_file, train_file, test_file)
    
    # Create Model with specified optimizer and loss function
    ##############################################################
    model = fastText(config, len(dataset.vocab), dataset.word_embeddings)
    # if torch.cuda.is_available():
    #     model.cuda()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=config.lr)
    NLLLoss = nn.NLLLoss()
    model.add_optimizer(optimizer)
    model.add_loss_op(NLLLoss)
    ##############################################################
    
    train_losses = []
    val_accuracies = []
    
    for i in range(config.max_epochs):
        print ("Epoch: {}".format(i))
        train_loss,val_accuracy = model.run_epoch(dataset.train_iterator, dataset.val_iterator, i)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)

    train_acc = evaluate_model(model, dataset.train_iterator)
    val_acc = evaluate_model(model, dataset.val_iterator)
    test_acc = evaluate_model(model, dataset.test_iterator)
    train_ori_acc = evaluate_model(model, dataset.train_val_iterator)

    print ('Final Training Accuracy: {:.4f}'.format(train_acc))
    print ('Final Validation Accuracy: {:.4f}'.format(val_acc))
    print ('Final Training + Validation Accuracy: {:.4f}'.format(train_ori_acc))
    print ('Final Test Accuracy: {:.4f}'.format(test_acc))

    # save model
    print('save model to model.pth')
    torch.save(model.state_dict(), 'model.pth')
    model.load_state_dict(torch.load('model.pth'))
    # save dadtaset.vocab and dataset.word_embeddings to pickle
    # with open('dataset.pkl', 'wb') as f:
    #     pickle.dump((dataset.vocab, dataset.word_embeddings), f)
