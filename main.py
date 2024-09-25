import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torchnet.meter as meter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_curve, precision_recall_curve,
                             roc_auc_score, average_precision_score, confusion_matrix)

import os
import time
import sys
import argparse
import csv
from pathlib import Path

from utils import MyData, progressBar, format_time, saveCheckpoint, saveMkdir, save_log_to_csv, LogMeters, calculate_equalized_odds_difference, calculate_disparate_impact
from model import FGAM

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

class Args:
    def __init__(self):
        self.num_workers = 4
        self.batch_size = 32
        self.epochs = 30
        self.embedding = 8
        self.nhid = 2
        self.batch_norm = True
        self.learning_rate = 0.01
        self.optimizer = 'Adam'
        self.prefix = ''
        self.use_gpu = True
        self.n_classes = 2
        self.dim_time_varying = 59
        self.dim_static = 27

def _train(epoch, data_loader, log, model, optimizer, dim_time_varying):
    model.train()
    log.reset()
    total_loss, total_batches = 0.0, 0.0
    for batch_idx, (static, time_variying, targets) in enumerate(iter(data_loader)):
        for idx in range(dim_time_varying):
            time_variying[idx] = time_variying[idx].cuda()
        static = static.cuda()
        targets = targets.squeeze().long().cuda()
        optimizer.zero_grad()
        outputs = F.log_softmax(model(static, time_variying), dim=1)
        log.update(outputs, targets)
        loss = F.nll_loss(outputs, targets)
        total_loss += loss
        total_batches += 1
        loss.backward()
        optimizer.step()
    print('Avg loss is {}'.format(total_loss / total_batches))
    log.printLog(epoch)

def _test(epoch, data_loader, log, model, dim_time_varying):
    model.eval()
    log.reset()
    for batch_idx, (static, time_variying, targets) in enumerate(iter(data_loader)):
        static = static.cuda()
        for idx in range(dim_time_varying):
            time_variying[idx] = time_variying[idx].cuda()
        targets = targets.squeeze().long().cuda()
        outputs = F.log_softmax(model(static, time_variying), dim=1)
        log.update(outputs, targets)
    
    return log.printLog(epoch)
    
def train(args, model, data_train, data_valid):
    log_tr = LogMeters(args.prefix + args.optimizer + '_Train', args.n_classes)
    log_tv = LogMeters(args.prefix + args.optimizer + '_Test', args.n_classes)

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters())
    elif args.optimizer == 'LBFGS':
        optimizer = optim.LBFGS(model.parameters())
    elif args.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters())
    elif args.optimizer == 'Rprop':
        optimizer = optim.Rprop(model.parameters())
    elif args.optimizer == 'Adadelta':
        optimizer = optim.Adadelta(model.parameters())
    else:
        raise ValueError('Invalid optimizer specified.')

    train_loader = DataLoader(data_train, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(data_valid, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)
    
    for epoch in range(args.epochs):
        print(f'--Traning epoch {epoch}')
        _train(epoch, train_loader, log_tr, model, optimizer, args.dim_time_varying)
        print('--Validation...')
        _test(epoch, valid_loader, log_tv, model, args.dim_time_varying)

        saveCheckpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': args,
        }, './log/'+args.prefix+args.optimizer+'_Test/fgam'+str(epoch)+'.pth.tar')
    
def test(args, data_test, model):
    test_loader = DataLoader(data_test, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)
    log_te = LogMeters(args.prefix + args.optimizer + '_Test', args.n_classes)

    return _test(0, test_loader, log_te, model, args.dim_time_varying)

def fairness_assessment(y_true, y_pred, data_test, groups):
    fairness_metrics = {}
    for group in groups:    
        # Pull the feature            
        sensitive_feature = data_test.X_test[group]

        # Compute fairness metrics       
        fairness_metrics[group] = [
            {'Equalized odds diff': calculate_equalized_odds_difference(y_true, y_pred, sensitive_feature)},
            {'Diaparate impact': calculate_disparate_impact(y_pred, sensitive_feature)}
        ]
    return fairness_metrics

def run(mode = 'test',
        model_type = 'reweighed',
        data_dir = ROOT / 'data',
        model_dir = ROOT/'model',
        ):
    np.random.seed(seed=42)
    plt.switch_backend('agg')
    args = Args()
        
    print('Loading dataset...')
    
    if model_type == 'base':
        data_path = str(data_dir / 'df_base.csv')
        model_path = str(model_dir / 'model_base.pth.tar')
    elif model_type == 'blinded':
        data_path = str(data_dir / 'df_blinded.csv')
        model_path = str(model_dir / 'model_blinded.pth.tar')
    else:
        data_path = str(data_dir / 'df_reweighed.csv')
        model_path = str(model_dir / 'model_reweighed.pth.tar')
    wt_path = str(data_dir / 'weights.csv')
    
    data_train = MyData(data_type='train', file_path=data_path)
    data_valid = MyData(data_type='valid', file_path=data_path)
    data_test = MyData(data_type='test', file_path=data_path, weight_path=wt_path)
    
    model = FGAM(args.n_classes, args.dim_time_varying, args.dim_static,
                 args.embedding, args.nhid, args.batch_norm)
    if args.use_gpu:
        model.cuda()
    
    if mode == 'train':
        train(args, model, data_train, data_valid)
    if mode == "test":
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        
        results = test(args, data_test, model)
        y_true = results[0].squeeze()
        y_scores = results[1].squeeze()
        y_pred = np.where(results[1].squeeze() >= 0.5, 1, 0)
        
        weights = data_test.wt_test
        
        # Post-processing performance
        precision = precision_score(y_true, y_pred, sample_weight=weights)
        recall = recall_score(y_true, y_pred, sample_weight=weights)
        f1 = f1_score(y_true, y_pred, sample_weight=weights)
        auroc = roc_auc_score(y_true, y_scores, sample_weight=weights)
        auprc = average_precision_score(y_true, y_scores, sample_weight=weights)
        print()
        print(f'weighted precision: {precision:.6f}, recall: {recall:.6f}, f1: {f1:.6f}, auroc: {auroc:.6f}, auprc: {auprc:.6f}')
        print()
        
        # Fairness assessment metrics
        ethnicity_columns = [col for col in data_test.X_test.columns if col.startswith('race')]
        insurance_columns = [col for col in data_test.X_test.columns if col.startswith('insurance')]
        icu_columns = [col for col in data_test.X_test.columns if col.startswith('icu_group')]
        data_test.X_test['Ethnicity'] = data_test.X_test[ethnicity_columns].idxmax(axis=1)
        data_test.X_test['Insurance'] = data_test.X_test[insurance_columns].idxmax(axis=1)
        data_test.X_test['ICU_group'] = data_test.X_test[icu_columns].idxmax(axis=1)
        data_test.X_test['days_after_icu_4d_or_more'] = np.where(data_test.X_test['days_after_icu'] < 4, 0, 1)
    
        metrics = fairness_assessment(y_true, y_pred, data_test, groups=['gender_num', 'Ethnicity', 'Insurance', 'ICU_group',
                                                                         'days_after_icu_4d_or_more', 'is_married'])
        for key, value in metrics.items():
            print(key)
            print(value)
        
        # Threshold adjustment to the two sensitive groups
        sensitive_feature_1 = ((data_test.X_test['icu_group_1.0'] == 1) | (data_test.X_test['icu_group_2.0'] == 1))
        sensitive_feature_2 = data_test.X_test['days_after_icu_4d_or_more']
        pred_group_1 = (y_scores[(sensitive_feature_1 == 0) & (sensitive_feature_2 == 0)] > 0.5).astype(int)
        pred_group_2 = (y_scores[(sensitive_feature_1 == 1) & (sensitive_feature_2 == 0)] > 0.75).astype(int)
        pred_group_3 = (y_scores[(sensitive_feature_1 == 0) & (sensitive_feature_2 == 1)] > 0.93).astype(int)
        pred_group_4 = (y_scores[(sensitive_feature_1 == 1) & (sensitive_feature_2 == 1)] > 0.97).astype(int)
        # Combine predictions back into the original shape
        y_pred_post = y_pred.copy()
        y_pred_post[(sensitive_feature_1 == 0) & (sensitive_feature_2 == 0)] = pred_group_1
        y_pred_post[(sensitive_feature_1 == 1) & (sensitive_feature_2 == 0)] = pred_group_2
        y_pred_post[(sensitive_feature_1 == 0) & (sensitive_feature_2 == 1)] = pred_group_3
        y_pred_post[(sensitive_feature_1 == 1) & (sensitive_feature_2 == 1)] = pred_group_4

        # Print post-adjustment metrics
        metrics = fairness_assessment(y_true, y_pred_post, data_test, groups=['ICU_group', 'days_after_icu_4d_or_more'])
        print()
        print('After threshold adjustment: ')
        for key, value in metrics.items():
            print(key)
            print(value)
            
        # Print post-adjustment performances
        precision = precision_score(y_true, y_pred_post)
        recall = recall_score(y_true, y_pred_post)
        f1 = f1_score(y_true, y_pred_post)
        print()
        print(f'Threshold adjusted precision: {precision:.6f}, recall: {recall:.6f}, f1: {f1:.6f}')
        print()
        
def parse_opt():      
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='test', help='train or test')
    parser.add_argument('--model_type', default='base', help='base, blinded, or reweighed')
    parser.add_argument('--data_dir', default=ROOT / 'data', help='data dir path')
    parser.add_argument('--model_dir', default=ROOT / 'model', help='model dir path')
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))
            
if __name__ == '__main__':
    opt = parse_opt()
    main(opt)