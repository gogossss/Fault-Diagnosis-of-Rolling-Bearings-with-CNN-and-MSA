# 自学PYTHON
# 开发时间 2023/9/30 9:03
import os
import random
import sys

import numpy as np
import torch
from sklearn import metrics, tree
from torch import nn, optim
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import preprocess
from preprocess import get_loader
from guoqing.model.my_model import MyBertModel, Proto, MY_CNN_LSTM


class MyModel:
    def __init__(self, sentence_encoder, random_seed=0):
        self.data_loader = get_loader
        self.random_seed = random_seed
        self.sentence_encoder = sentence_encoder

    def __load_model__(self, ckpt):
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)

    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    def run(self,
            classifier,
            path,
            save_ckpt=None,
            N=5, K=1, Q=1,
            batch_size=1,
            length=784,
            number=300,
            normal=True,
            is_train=True,
            rate=[0.5, 0.25, 0.25],
            trainN=5):

        np.random.seed(self.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        random.seed(self.random_seed)

        x_train, y_train, x_valid, y_valid, x_test, y_test, train_data, test_data, val_data = preprocess.prepro(
            d_path=path,
            length=length,
            number=number,
            normal=normal,
            rate=rate,
            enc=False, enc_step=28, random_seed=self.random_seed)

        model = classifier
        if model == 'proto':
            if self.sentence_encoder == 'CNN_LSTM':
                encoder_name = 'bert'
            else:
                encoder_name = 'cnn_lstm'
            print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
            print("model: {}".format(model))
            print("encoder: {}".format(encoder_name))


            prefix = '-'.join([model, encoder_name, str(N), str(K)])
            if self.sentence_encoder == 'CNN_LSTM':
                model = Proto(MY_CNN_LSTM(in_channels=length, out_channels=32, hidden_size=128, num_layers=2, output_size=200, batch_size=batch_size))
            else:
                model = Proto(self.sentence_encoder)

            if not os.path.exists('checkpoint'):
                os.mkdir('checkpoint')
            ckpt = 'checkpoint/{}.pth.tar'.format(prefix)
            if save_ckpt:
                ckpt = save_ckpt
            if torch.cuda.is_available():
                model.cuda()

            if is_train == True:
                self.train(model, prefix, train_data, test_data, val_data, batch_size, trainN, N, K, Q, save_ckpt=ckpt)
                load_ckpt = ckpt
            else:
                load_ckpt = 'none'

            acc = self.eval(model, test_data, val_data, batch_size, N, K, Q, ckpt=load_ckpt)
            print("RESULT: %.2f" % (acc * 100))


        else:
            model.fit(x_train, y_train)
            pred_test = model.predict(x_test)
            score = metrics.accuracy_score(y_test, pred_test)
            print('accuracy_score', score)

    def train(self,
              model,
              model_name,
              train_data, test_data, val_data,
              B, N_for_train, N_for_eval, K, Q,
              na_rate=0,
              learning_rate=1e-5,
              lr_step_size=20000,
              weight_decay=1e-5,
              train_iter=30000,
              val_iter=50,
              val_step=500,
              test_iter=3000,
              load_ckpt=None,
              save_ckpt=None,
              pytorch_optim=optim.SGD,
              warmup=True,
              warmup_step=300,
              grad_iter=1,
              pair=False,
              use_sgd_for_bert=False):

        print("Start training...")

        # Init
        print('Use bert optim!')
        parameters_to_optimize = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in parameters_to_optimize
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if use_sgd_for_bert:
            optimizer = torch.optim.SGD(parameters_to_optimize, lr=learning_rate)
        else:
            optimizer = AdamW(parameters_to_optimize, lr=learning_rate, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step,
                                                    num_training_steps=train_iter)

        if load_ckpt:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    print('ignore {}'.format(name))
                    continue
                print('load {} from {}'.format(name, load_ckpt))
                own_state[name].copy_(param)
            start_iter = 0
        else:
            start_iter = 0

        model.train()

        # Training
        best_acc = 0
        iter_loss = 0.0
        iter_loss_dis = 0.0
        iter_right = 0.0
        iter_right_dis = 0.0
        iter_sample = 0.0
        for it in range(start_iter, start_iter + train_iter):
            if self.sentence_encoder == 'CNN_LSTM':
                sentence_encoder = MY_CNN_LSTM(in_channels=784, out_channels=40, hidden_size=784, num_layers=2, output_size=128, batch_size=B)
            else:
                sentence_encoder = self.sentence_encoder

            support, query, label = next(self.data_loader(train_data, sentence_encoder,
                N=N_for_train, K=K, Q=Q, batch_size=B))
            if torch.cuda.is_available():
                for k in support:
                    support[k] = support[k].cuda()
                for k in query:
                    query[k] = query[k].cuda()
                label = label.cuda()
            logits, pred = model(support, query, N_for_train, K, Q * N_for_train + na_rate * Q)
            loss = model.loss(logits, label) / float(grad_iter)
            right = model.accuracy(pred, label)

            loss.requires_grad_(True)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

            if it % grad_iter == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            iter_loss += self.item(loss.data)
            iter_right += self.item(right.data)
            iter_sample += 1
            print('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(it + 1, iter_loss / iter_sample,
                                                                               100 * iter_right / iter_sample) + '\r')
            # sys.stdout.write(
            #     'step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(it + 1, iter_loss / iter_sample,
            #                                                                    100 * iter_right / iter_sample) + '\r')
            # sys.stdout.flush()

            if (it + 1) % val_step == 0:
                acc = self.eval(model, test_data, val_data, B, N_for_eval, K, Q, val_iter)
                model.train()
                if acc > best_acc:
                    print('Best checkpoint')
                    torch.save({'state_dict': model.state_dict()}, save_ckpt)
                    best_acc = acc
                iter_loss = 0.
                iter_loss_dis = 0.
                iter_right = 0.
                iter_right_dis = 0.
                iter_sample = 0.

        print("\n####################\n")
        print("Finish training " + model_name)

    def eval(self,
             model,
             test_data, val_data,
             B, N, K, Q,
             eval_iter=200,
             na_rate=0,
             ckpt=None
             ):

        print("")

        model.eval()
        if ckpt is None:
            print("Use val dataset")
            eval_dataset = val_data
        else:
            print("Use test dataset")
            if ckpt != 'none':
                state_dict = self.__load_model__(ckpt)['state_dict']
                own_state = model.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                        continue
                    own_state[name].copy_(param)
            eval_dataset = test_data

        iter_right = 0.0
        iter_sample = 0.0
        with torch.no_grad():
            for it in range(eval_iter):
                if self.sentence_encoder == 'CNN_LSTM':
                    sentence_encoder = MY_CNN_LSTM(in_channels=784, out_channels=32, hidden_size=784, num_layers=2,
                                                   output_size=128, batch_size=B)
                else:
                    sentence_encoder = self.sentence_encoder
                support, query, label = next(self.data_loader(eval_dataset, sentence_encoder,
                N=N, K=K, Q=Q, batch_size=B))
                if torch.cuda.is_available():
                    for k in support:
                        support[k] = support[k].cuda()
                    for k in query:
                        query[k] = query[k].cuda()
                    label = label.cuda()
                logits, pred = model(support, query, N, K, Q * N + Q * na_rate)

                right = model.accuracy(pred, label)
                iter_right += self.item(right.data)
                iter_sample += 1
                print('[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * iter_right / iter_sample) + '\r')
                # sys.stdout.write(
                #     '[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * iter_right / iter_sample) + '\r')
                # sys.stdout.flush()
            print("")
        return iter_right / iter_sample

if __name__ == '__main__':
    # MyModel(sentence_encoder=MyBertModel('bert-base-uncased', 128)).run(classifier='proto', path=r'D:\python\guoqing\data')
    MyModel(sentence_encoder='CNN_LSTM').run(classifier='proto', path=r'D:\python\guoqing\data')
