
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import random
import json
from utils import get_dataset, evaluation, get_network, get_logger, get_metrics, plot_series

def argparser():
    parser = argparse.ArgumentParser(description='centralized sgd baseline')

    # default args - data set and model
    parser.add_argument('--dataset', type=str, default='cifar10', help='choose datset')
    parser.add_argument('--model', type=str, default='ResNet18', help='model')
   
    # default args - experiments
    parser.add_argument('--seed', type=int, default=42, help='set a seed for reproducability')
    parser.add_argument('--num_exp', type=int, default=3, help='the number of experiment runs')  
    parser.add_argument('--n_workers', type=int, default=0, help='num of worksers for dataloader')

    # args - server
    parser.add_argument('--lr', type=float, default=0.01, help='server learning rate for updating global model by the server')
    parser.add_argument('--batch_train', type=int, default=64, help='server batch size for training global model')
    parser.add_argument('--epoch_train', type=int, default=50, help='server epochs to train global model with synthetic data')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='value of weight decay for optimizer')

    # args - results
    parser.add_argument('--save_root', type=str, default='result', help='path to save results')
    args = parser.parse_args()

    return args

def set_path(args):
    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)

    # prepare the save path
    save_tag = f'centralized_sgd-{args.dataset}-{args.model}-ep{args.epoch_train}-lr{args.lr}' 

    # if args.save_results or args.save_curves:
    exp_seq_path = os.path.join(args.save_root, 'exp_seq.txt')
    if not os.path.exists(exp_seq_path):
        file = open(exp_seq_path, 'w')
        exp_seq=0
        exp_seq = str(exp_seq)
        file.write(exp_seq)
        file.close
        save_tag = 'exp_' + exp_seq + '_' + save_tag
    else:
        file = open(exp_seq_path, 'r')
        exp_seq = int(file.read())
        exp_seq += 1
        exp_seq = str(exp_seq)
        save_tag = 'exp_' + exp_seq + '_' + save_tag
        file = open(exp_seq_path, 'w')
        file.write(exp_seq)
        file.close()
    args.exp_seq = exp_seq
    args.save_path = os.path.join(args.save_root, save_tag)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    args.config_path = os.path.join(args.save_path, 'config.json')
    args.logger_path = os.path.join(args.save_path, 'exp_log.log')   
    # args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
   
    return args

def main(args, logger):   

    # show and save hyperparameter configuration
    with open(args.config_path, 'w') as f:
        f.write(json.dumps(args.__dict__, indent=4))
        f.close()
    
    ds, data_info = get_dataset(dataset=args.dataset)
    logger.info('{} dataset is loaded: {:6d} train data, {:6d} test data'.format(args.dataset, len(ds['train_data']), len(ds['test_data'])))
    
    # record performance for all experiment runs
    acc_all_exps = [[] for i in range(args.num_exp)]
    auc_all_exps = [[] for i in range(args.num_exp)]
    loss_all_exps = [[] for i in range(args.num_exp)]

    # looping over multiple experiment trials
    for exp in range(args.num_exp):
        seed = int(args.seed + exp)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)    
        logger.info('================== Exp {} =================='.format(exp))       

        
        # set the architecture for the network to be trained
        if args.model.lower() == 'resnet18':
            # net_train = get_network('ResNet18BN', data_info['channel'], data_info['num_classes'], data_info['img_size']).to(args.device)
            net_train = torchvision.models.resnet18(num_classes=data_info['num_classes']).to(args.device)
        if args.model.lower() == 'resnet50':
            # net_train = get_network('ResNet18BN', data_info['channel'], data_info['num_classes'], data_info['img_size']).to(args.device)
            net_train = torchvision.models.resnet50(num_classes=data_info['num_classes']).to(args.device)
        elif args.model.lower() == 'convnet':
            net_train = get_network('ConvNetBN', data_info['channel'], data_info['num_classes'], data_info['img_size']).to(args.device)
        logger.info('Model initialization completed')     
        
        train_loader = DataLoader(ds['train_data'], batch_size=args.batch_train, shuffle=True, num_workers=args.n_workers)
        test_loader = DataLoader(ds['test_data'], batch_size=2*args.batch_train, shuffle=False, num_workers=args.n_workers)
        optimizer = torch.optim.SGD(net_train.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)  # optimizer_img for synthetic data
        criterion = nn.CrossEntropyLoss().to(args.device)
        
        best_loss, best_acc, best_auc = 1e10, 0.0, 0.0
        for i in range(args.epoch_train):
            train_loss, n_samples = 0.0, 0
            net_train.train()
            for x, y in train_loader:
                x, y = x.to(args.device), y.to(args.device)
                output = net_train(x)
                loss = criterion(output,y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * y.size(0)
                n_samples += y.size(0)  
            
            # test_acc, test_loss = evaluation(args, net_train, test_loader) # use either evaluation() or get_metrics()
            test_loss, test_acc, test_auc, _ = get_metrics(args, net_train, test_loader)
            loss_all_exps[exp].append(test_loss)
            acc_all_exps[exp].append(test_acc)
            auc_all_exps[exp].append(test_auc)
                        
            if test_loss<best_loss:
                best_loss = test_loss
            
            if test_acc>best_acc:
                best_acc = test_acc
                torch.save({
                    'epoch': i,
                    'model_state_dict': net_train.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': criterion,
                    }, os.path.join(args.save_path, 'checkpoint.pt'))
                logger.info(f'Epoch {i:3d}: checkpoint saved')

            if test_acc>best_auc:
                best_auc = test_auc

            logger.info(f'>> Epoch {i:3d}: test loss = {test_loss:.6f}, running best = {best_loss:.6f}')
            logger.info(f'>> Epoch {i:3d}: test acc = {test_acc*100:.2f}%, running best = {best_acc*100:.2f}%')
            logger.info(f'>> Epoch {i:3d}: test auc = {test_auc:.4f}, running best = {best_auc:.4f}')

    # summary and stats of all experiment trials
    loss_all_exps = np.array(loss_all_exps)
    acc_all_exps = np.array(acc_all_exps)
    auc_all_exps = np.array(auc_all_exps)
    acc_max = acc_all_exps.max(axis=1)
    auc_max = auc_all_exps.max(axis=1)
    logger.info('>> Mean performance of the global model, averaged over {} experiments: '.format(args.num_exp))
    logger.info('>> Maximum acc = {:.2f}%, std = {:.2f}%'.format(np.mean(acc_max)*100, np.std(acc_max)*100))
    logger.info('>> Maximum auc = {:.4f}, std = {:.4f}'.format(np.mean(auc_max), np.std(auc_max)))
    
    # plot and save test loss curves
    plot_series(
        series=loss_all_exps.mean(axis=0),
        y_min=0,
        y_max=np.ceil(loss_all_exps.mean(axis=0).max()),
        series_name='Test loss',
        title=f'Test loss training {args.model} on {args.dataset}',
        save_path=os.path.join(args.save_path, f'loss_{args.model}_{args.model}.png')
    )

    # plot and save test acc curves
    plot_series(
        series=acc_all_exps.mean(axis=0),
        y_min=0,
        y_max=1.0,
        series_name='Test acc',
        title=f'Test acc training {args.model} on {args.dataset}',
        save_path=os.path.join(args.save_path, f'acc_{args.model}_{args.model}.png')
    )

    # plot and save test auc curves
    plot_series(
        series=auc_all_exps.mean(axis=0),
        y_min=0,
        y_max=1.0,
        series_name='Test auc',
        title=f'Test auc training {args.model} on {args.dataset}',
        save_path=os.path.join(args.save_path, f'auc_{args.model}_{args.model}.png')
    )

    torch.save(
        {
            'test_loss': loss_all_exps,
            'test_acc': acc_all_exps,
            'test_auc': auc_all_exps
        },
        os.path.join(args.save_path, 'learning_curve_{}_{}.pt'.format(args.dataset, args.model))
    )

if __name__ == '__main__':
    args = argparser()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    set_path(args)
    logger = get_logger(args.logger_path)
    
    time_start = time.time()
    main(args, logger)    
    time_end = time.time()
    
    time_end_stamp = time.strftime('%Y-%m-%d %H:%M:%S') # time_end_stamp = time.strftime('%y-%m-%d-%H-%M-%S')
    sesseion_time = int((time_end-time_start)/60)   
    print('-------------------------\nSession: Exp {} completed at {}, time elapsed: {} mins. That\'s all folks.'.format(args.exp_seq,time_end_stamp, sesseion_time))
    



