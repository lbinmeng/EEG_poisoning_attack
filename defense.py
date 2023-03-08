import os
import logging
import torch
import argparse
import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, dataset
from models import LoadModel
from utils.pytorch_utils import init_weights, print_args, seed, weight_for_balanced_classes, bca_score, split_data
from utils.data_loader import load


def train(x_train: torch.Tensor, y_train: torch.Tensor,
          x_valiation: torch.Tensor, y_validation: torch.Tensor, x_test: torch.Tensor, 
          x_test_poison: torch.Tensor, y_test: torch.Tensor, args):
    # initialize the model
    model = LoadModel(args.model, 
                      n_classes=len(np.unique(y_train.numpy())), 
                      Chans=x_train.shape[2], 
                      Samples=x_train.shape[3],
                      spa_frac=args.spa_frac)
    model.to(args.device)
    model.apply(init_weights)

    # trainable parameters
    params = []
    for _, v in model.named_parameters():
        params += [{'params': v, 'lr': args.lr}]
    optimizer = optim.Adam(params, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss().to(args.device)

    # data loader
    # for unbalanced dataset, create a weighted sampler
    # sample_weights = weight_for_balanced_classes(y_train)
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(
    #     sample_weights, len(sample_weights))
    # train_loader = DataLoader(dataset=TensorDataset(x_train, y_train),
    #                           batch_size=args.batch_size,
    #                           sampler=sampler,
    #                           drop_last=False)
    # validation_loader = DataLoader(dataset=TensorDataset(x_valiation, y_validation),
    #                           batch_size=args.batch_size,
    #                           shuffle=True,
    #                           drop_last=False)
    train_loader = DataLoader(dataset=TensorDataset(x_train, y_train),
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=False)
    test_loader = DataLoader(dataset=TensorDataset(x_test, x_test_poison, y_test),
                             batch_size=args.batch_size,
                             shuffle=True,
                             drop_last=False)

    best_val_bca, earlystop_cnt, best_model = 0., 0, None
    for epoch in range(args.epochs):
        # model training
        adjust_learning_rate(optimizer=optimizer, epoch=epoch + 1, args=args)
        model.train()
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            model.MaxNormConstraint()

        # validation for early stopping
        # model.eval()
        # _, _, validation_bca = eval(model, criterion, validation_loader, args)
        # if validation_bca > best_val_bca:
        #     best_val_bca = validation_bca
        #     best_model = copy.deepcopy(model)
        #     earlystop_cnt = 0
        # else:
        #     earlystop_cnt += 1
        #     if earlystop_cnt > args.patience:
        #         logging.info(f'Early stopping in epoch {epoch}')
        #         break

        if (epoch + 1) % 10 == 0:
            model.eval()
            train_loss, train_acc, train_bca = eval(model, criterion, train_loader, args)
            test_acc, test_bca, asr = peval(model, test_loader, args)

            logging.info(
                'Epoch {}/{}: train loss: {:.4f} train acc: {:.2f} train bca: {:.2f}| test acc: {:.2f} test bca: {:.2f} ASR: {:.2f}'
                .format(epoch + 1, args.epochs, train_loss, train_acc, train_bca, test_acc, test_bca, asr))

    # fine-tuning defense
    if args.defense == 'finetuning':
        finetuning_num = int(0.1 * len(x_test))
        idx = np.random.permutation(np.arange(len(x_test)))
        sample_weights = weight_for_balanced_classes(y_test)
        sample_weights = np.array(sample_weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights[idx[:finetuning_num]], finetuning_num)
        finetuning_loader = DataLoader(dataset=TensorDataset(x_test[idx[:finetuning_num]], y_test[idx[:finetuning_num]]),
                             batch_size=args.batch_size,
                             sampler=sampler,
                             drop_last=False)
        params = []
        for _, v in model.named_parameters():
            params += [{'params': v, 'lr': 1e-3}]
        optimizer = optim.Adam(params, weight_decay=5e-4)
        for epoch in range(50):
            # model training
            model.train()
            for step, (batch_x, batch_y) in enumerate(finetuning_loader):
                batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
                optimizer.zero_grad()
                out = model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                model.MaxNormConstraint()
        
    model.eval()
    test_acc, test_bca, asr = peval(model, test_loader, args)
    logging.info(f'test bca: {test_bca} ASR: {asr}')

    return test_acc, test_bca, asr


def eval(model: nn.Module, criterion: nn.Module, data_loader: DataLoader, args):
    loss, correct = 0., 0
    labels, preds = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(args.device), y.to(args.device)
            out = model(x)
            pred = nn.Softmax(dim=1)(out).cpu().argmax(dim=1)
            loss += criterion(out, y).item()
            correct += pred.eq(y.cpu().view_as(pred)).sum().item()
            labels.extend(y.cpu().tolist())
            preds.extend(pred.tolist())
    loss /= len(data_loader.dataset)
    acc = correct / len(data_loader.dataset)
    bca = bca_score(labels, preds)

    return loss, acc, bca

def peval(model: nn.Module, data_loader: DataLoader, args):
    correct = 0
    labels, preds, ppreds = [], [], []
    with torch.no_grad():
        for x, px, y in data_loader:
            x, px, y = x.to(args.device), px.to(args.device), y.to(args.device)
            out = model(x)
            pout = model(px)
            pred = nn.Softmax(dim=1)(out).cpu().argmax(dim=1)
            ppred = nn.Softmax(dim=1)(pout).cpu().argmax(dim=1)
            correct += pred.eq(y.cpu().view_as(pred)).sum().item()
            labels.extend(y.cpu().tolist())
            preds.extend(pred.tolist())
            ppreds.extend(ppred.tolist())
        acc = correct / len(data_loader.dataset)
        bca = bca_score(labels, preds)
        valid_idx = [x for x in range(len(labels)) if labels[x]==preds[x] and labels[x] != args.target_label]
        if len(valid_idx) == 0: asr = np.nan
        else:
            asr = len([x for x in valid_idx if ppreds[x]==args.target_label]) / len(valid_idx)
    return  acc, bca, asr


def adjust_learning_rate(optimizer: nn.Module, epoch: int, args):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 50:
        lr = args.lr * 0.1
    if epoch >= 150:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model train')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--model', type=str, default='EEGNet')
    parser.add_argument('--dataset', type=str, default='ERN')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--target_label', type=int, default=1)
    parser.add_argument('--a', type=float, default=0.3, help='NPP amplitude')
    parser.add_argument('--f', type=int, default=5, help='NPP freq')
    parser.add_argument('--p', type=float, default=0.1, help='NPP proportion')
    parser.add_argument('--pr', type=float, default=0.05, help='poison_rate')
    parser.add_argument('--baseline', type=bool, default=False, help='is baseline')
    parser.add_argument('--physical', type=bool, default=False, help='is physical')
    parser.add_argument('--partial', type=float, default=None,  help='partial rate')
    parser.add_argument('--defense', type=str, default='finetuning', help='defense type') # SAP finetuning

    args = parser.parse_args()

    args.device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    subject_numbers = {'ERN': 16, 'P300': 8, 'MI4C': 9}
    npp_params = [args.a, args.f, args.p]
    subject_number = subject_numbers[args.dataset]
    downsample = False if args.dataset == 'MI4C' else True

    # path build
    log_path = f'results_revision_new/log/defense/{args.defense}/'
    log_path += f'{args.dataset}/{args.model}/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = log_path + ('baseline_' if args.baseline else 'npp_') + f'{args.a}_{args.f}_{args.p}.log'

    npz_path = f'results_revision_new/npz/defense/{args.defense}/' 
    npz_path += f'{args.dataset}/{args.model}/'
    if not os.path.exists(npz_path):
        os.makedirs(npz_path)
    npz_name = npz_path + ('baseline_' if args.baseline else 'npp_') + f'{args.a}_{args.f}_{args.p}.npz'

    if args.partial: 
        log_name = log_name.replace('.log', f'_{args.partial}.log')
        npz_name = npz_name.replace('.npz', f'_{args.partial}.npz')
    
    args.spa_frac = (0.9 if args.defense == 'SAP' else None)

    # logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # print logging
    print_log = logging.StreamHandler()
    logger.addHandler(print_log)
    # save logging
    save_log = logging.FileHandler(log_name, mode='w', encoding='utf8')
    logger.addHandler(save_log)

    logging.info(print_args(args) + '\n')
    

    raccs, rbcas, rasrs = [], [], []
    for r in range(10):
        seed(r)
        accs, bcas, asrs = [], [], []
        s_id = np.random.permutation(np.arange(subject_number))
        for s in range(1, subject_number):
            x_train, y_train = [], []
            train_idx = [x for x in range(0, subject_number)]
            train_idx.remove(s_id[0])
            train_idx.remove(s_id[s])
            for i in train_idx:
                _, x_i, y_i = load(args.dataset, i, npp_params, clean=True, physical=args.physical, downsample=downsample)
                x_train.append(x_i)
                y_train.append(y_i)
            x_train = np.concatenate(x_train, axis=0)
            y_train = np.concatenate(y_train, axis=0)

            # x_train, y_train, x_validation, y_validation = split_data([x_train, y_train], split=0.8, shuffle=True)

            # create poison data 
            _, x_p, y_p = load(args.dataset, s_id[0], npp_params, clean=False, physical=args.physical, partial=args.partial, downsample=False)
            idx = np.random.permutation(np.arange(len(x_p)))
            x_poison, y_poison = x_p[idx[:int(args.pr * len(x_train))]], y_p[idx[:int(args.pr * len(x_train))]]
            y_poison = np.ones(shape=y_poison.shape) * args.target_label # target label

            if not args.baseline:
                x_train = np.concatenate([x_train, x_poison], axis=0)
                y_train = np.concatenate([y_train, y_poison], axis=0)

            # leave one subject validation
            _, x_test, y_test = load(args.dataset, s_id[s], npp_params, clean=True, physical=args.physical, downsample=False)
            _, x_test_poison, _ = load(args.dataset, s_id[s], npp_params, clean=False, physical=args.physical, partial=args.partial, downsample=False)

            logging.info(f'train: {x_train.shape}, test: {x_test.shape}')
            x_train = Variable(
                torch.from_numpy(x_train).type(torch.FloatTensor))
            y_train = Variable(
                torch.from_numpy(y_train).type(torch.LongTensor))
            # x_validation = Variable(
            #     torch.from_numpy(x_validation).type(torch.FloatTensor))
            # y_validation = Variable(
            #     torch.from_numpy(y_validation).type(torch.LongTensor)) 
            x_test = Variable(torch.from_numpy(x_test).type(torch.FloatTensor))
            x_test_poison = Variable(torch.from_numpy(x_test_poison).type(torch.FloatTensor))
            y_test = Variable(torch.from_numpy(y_test).type(torch.LongTensor))

            acc, bca, asr = train(x_train, y_train, None, None, x_test, x_test_poison, y_test, args)
            accs.append(acc)
            bcas.append(bca)
            asrs.append(asr)

        logging.info(f'Mean ACC: {np.nanmean(accs)}, BCA: {np.nanmean(bcas)}, ASR: {np.nanmean(asrs)}')
        raccs.append(accs)
        rbcas.append(bcas)
        rasrs.append(asrs)

    logging.info(f'ACCs: {np.nanmean(raccs, 1)}')
    logging.info(f'BCAs: {np.nanmean(rbcas, 1)}')
    logging.info(f'ASRs: {np.nanmean(rasrs, 1)}')
    logging.info(f'ALL ACC: {np.nanmean(raccs)} BCA: {np.nanmean(rbcas)} ASR: {np.nanmean(rasrs)}')
    np.savez(npz_name, raccs=raccs, rbcas=rbcas, rasrs=rasrs)
