import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchnet.meter as meter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from scipy import io

import os
import time
import sys

class MyData(Dataset):
    def __init__(self, data_type='train', file_path='reordered.csv', weight_path=None):
        assert data_type in ('train', 'valid', 'test')

        # Load the data from CSV
        full_data = pd.read_csv(file_path)

        u = 27
        m = len(full_data.columns) - 1
        
        # Assume the first 19 columns are unmodifiable features,
        # the next 51 columns are modifiable features, and the last column is the label
        features = full_data.iloc[:, :-1]  # 70 columns
        labels = full_data.iloc[:, -1]      # 마지막 열은 레이블

        # Split the data into train, valid, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(
            features, labels, test_size=0.2, stratify=labels, shuffle=True, random_state=42
        )
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, shuffle=True, random_state=42
        )
        if weight_path:
            wt = pd.read_csv(weight_path)
            # Split the data into train, valid, and test sets
            X_train, X_temp, y_train, y_temp, wt_train, wt_temp = train_test_split(
                features, labels, wt, test_size=0.2, stratify=labels, shuffle=True, random_state=42
            )
            X_valid, X_test, y_valid, y_test, wt_valid, wt_test = train_test_split(
                X_temp, y_temp, wt_temp, test_size=0.5, stratify=y_temp, shuffle=True, random_state=42
            )
            self.wt_train = wt_train
            self.wt_test = wt_test
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        # Assign data based on the type
        if data_type == 'train':
            self.unmodifiable_features = X_train.iloc[:, :u].values.astype(np.float32)
            self.modifiable_features = X_train.iloc[:, u:m].values.astype(np.float32)
            self.labels = y_train.values.astype(np.int32).reshape(-1, 1)
        elif data_type == 'valid':
            self.unmodifiable_features = X_valid.iloc[:, :u].values.astype(np.float32)
            self.modifiable_features = X_valid.iloc[:, u:m].values.astype(np.float32)
            self.labels = y_valid.values.astype(np.int32).reshape(-1, 1)
        else:  # test
            self.unmodifiable_features = X_test.iloc[:, :u].values.astype(np.float32)
            self.modifiable_features = X_test.iloc[:, u:m].values.astype(np.float32)
            self.labels = y_test.values.astype(np.int32).reshape(-1, 1)

        self.data_size = self.labels.shape[0]

    def __getitem__(self, index):
        unmodifiable_feature = self.unmodifiable_features[index]
        modifiable_feature = self.modifiable_features[index]
        label = self.labels[index]
        return unmodifiable_feature, [torch.from_numpy(np.array([ele])) for ele in modifiable_feature], label
        #return torch.from_numpy(unmodifiable_feature).astype(torch.int32), [torch.from_numpy(np.array([ele])) for ele in modifiable_feature], label

    def __len__(self):
        return self.data_size

term_width = int(81)
TOTAL_BAR_LENGTH = 60
last_time = time.time()
begin_time = last_time

def progressBar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def saveCheckpoint(state, filename, is_best=False):
    torch.save(state, filename)
    if is_best:
        torch.save(state, 'best_'+filename)

def saveMkdir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass

def save_log_to_csv(epoch, log, filepath):
    """로그 데이터를 CSV 파일로 저장하는 함수"""
    with open(filepath, mode='a', newline='') as file:
        writer = csv.writer(file)
        if epoch == 0:  # 첫 에포크일 때 헤더 추가
            writer.writerow(['Epoch', 'AUC', 'APR', 'ACC'])
        writer.writerow([epoch, log.AUC, log.APR, log.ACC])

class LogMeters(object):
    def __init__(self, name=None, n_classes=2):
        self.name = name
        self.n_classes = n_classes
        self.path = os.path.join('log', name)

        os.makedirs(self.path, exist_ok=True)

        self.conf_mtr = meter.ConfusionMeter(n_classes)
        self.auc_mtr = meter.AUCMeter()
        self.err_mtr = meter.ClassErrorMeter(topk=[1], accuracy=True)
        saveMkdir(self.path)

        self.fp = open(os.path.join(self.path, 'res.log'), 'w')
        self.y_scores = np.array([], dtype=np.float32).reshape(0, 1)
        self.y_true = np.array([], dtype=np.float32).reshape(0, 1)

    def update(self, output, target):
        preds = output.data
        probs = torch.exp(preds)
        _, predicted = torch.max(probs, 1)
        self.conf_mtr.add(predicted, target.data)
        if self.n_classes == 2:
            self.auc_mtr.add(probs[:, 1],
                             target.data)
            curr_output = probs[:, 1].cpu().squeeze().numpy()
            curr_output.resize(curr_output.shape[0], 1)
            curr_target = target.data.cpu().squeeze().numpy()
            curr_target.resize(curr_target.shape[0], 1)
            self.y_scores = np.vstack([self.y_scores, curr_output])
            self.y_true = np.vstack([self.y_true, curr_target])

        self.err_mtr.add(probs, target.data)

    def printLog(self, epoch=0):
        conf_mtrx = self.conf_mtr.value()
        print(conf_mtrx)
        if self.n_classes == 2:
            conf_matrix = self.conf_mtr.value()
            tp = conf_matrix[1][1]  # True Positives
            fp = conf_matrix[0][1]  # False Positives
            fn = conf_matrix[1][0]  # False Negatives
            tn = conf_matrix[0][0]  # True Negatives
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            print(f'\tPrecision is {precision:.6f}')
            print(f'\tRecall is {recall:.6f}')
            print(f'\tf1 is {f1_score:.6f}')
    
            val_auc = roc_auc_score(self.y_true, self.y_scores)
            print('\tAUC is {:.6f}'.format(val_auc))
            average_precision = average_precision_score(self.y_true,
                                                        self.y_scores)
            print('\tAPR is {:.6f}'.format(average_precision))
            precision, recall, _ = precision_recall_curve(self.y_true,
                                                          self.y_scores)
            np.savetxt(self.path+'/precision_'+str(epoch)+'.txt',
                       precision, delimiter=',')
            np.savetxt(self.path+'/recall_'+str(epoch)+'.txt',
                       recall, delimiter=',')
            np.savetxt(self.path+'/true_'+str(epoch)+'.txt',
                       self.y_true, delimiter=',')
            np.savetxt(self.path+'/pred_'+str(epoch)+'.txt',
                       self.y_scores, delimiter=',')
            fpr, tpr, _ = roc_curve(self.y_true, self.y_scores)
            np.savetxt(self.path+'/fpr_'+str(epoch)+'.txt',
                       fpr, delimiter=',')
            np.savetxt(self.path+'/tpr_'+str(epoch)+'.txt',
                       tpr, delimiter=',')
        acc = self.err_mtr.value()
        acc = acc[0]
        print('\tACC is {:.6f}'.format(acc))
        self.fp.writelines('Confusion Matrix for ' + self.name+'\n')
        self.fp.writelines(str(conf_mtrx)+'\n')
        self.fp.writelines('AUC is {:.4f}'.format(val_auc)+'\n')
        self.fp.writelines('APR is {:.4f}'.format(average_precision)+'\n')
        self.fp.writelines('ACC Rate is {:.4f}%'.format(acc)+'\n')
        self.fp.writelines('\n')
        self.fp.flush()

        # plot image
        fig = plt.figure()
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.0])
        plt.xlim([0.0, 1.0])
        plt.title('Mortality Precision-Recall curve: AP={0:0.4f}'.format(
                      average_precision))
        fig.savefig(self.path+'/precision_recall_curve_' +
                    str(epoch) + '.pdf')
        plt.close(fig)

        fig = plt.figure()
        plt.step(fpr, tpr, color='b', alpha=0.2, where='post')
        plt.fill_between(fpr, tpr, step='post', alpha=0.2, color='b')

        plt.xlabel('False Postive Rate')
        plt.ylabel('True Positive Rate')
        plt.ylim([0.0, 1.0])
        plt.xlim([0.0, 1.0])
        plt.title('Mortality ROC curve: AUC={0:0.4f}'.format(val_auc))

        fig.savefig(self.path+'/ROC_curve_' + str(epoch) + '.pdf')
        plt.close(fig)
        
        return self.y_true, self.y_scores

    def reset(self):
        self.y_scores = np.array([]).reshape(0, 1)
        self.y_true = np.array([]).reshape(0, 1)
        self.conf_mtr.reset()
        self.auc_mtr.reset()
        self.err_mtr.reset()
        
        
def calculate_equalized_odds_difference(y_true, y_pred, sensitive_feature):
    # Initialize dictionaries to store TPR and FPR for each group
    tpr_group = {}
    fpr_group = {}
    
    # Identify unique groups in the sensitive feature
    unique_groups = np.unique(sensitive_feature)
    
    for group in unique_groups:
        # Filter predictions and true labels by sensitive group
        idx = (sensitive_feature == group)
        y_true_group = y_true[idx]
        y_pred_group = y_pred[idx]
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
        
        # Calculate TPR and FPR for this group
        tpr_group[group] = tp / (tp + fn)  # True Positive Rate
        fpr_group[group] = fp / (fp + tn)  # False Positive Rate
    
    # Calculate differences in TPR and FPR between groups
    tpr_diff = max(tpr_group.values()) - min(tpr_group.values())
    fpr_diff = max(fpr_group.values()) - min(fpr_group.values())
    
    equalized_odds_difference = max(tpr_diff, fpr_diff)
    
    return equalized_odds_difference

def calculate_disparate_impact(y_pred, sensitive_feature):
    # Initialize dictionary to store positive outcome rates for each group
    positive_rate_group = {}
    
    # Identify unique groups in the sensitive feature
    unique_groups = np.unique(sensitive_feature)
    
    for group in unique_groups:
        # Filter predictions by sensitive group
        idx = (sensitive_feature == group)
        y_pred_group = y_pred[idx]
        
        # Calculate positive outcome rate for this group
        positive_rate_group[group] = np.mean(y_pred_group == 1)
    
    # Calculate Disparate Impact as the ratio of the min positive rate to max positive rate
    disparate_impact = min(positive_rate_group.values()) / max(positive_rate_group.values())
    
    return disparate_impact