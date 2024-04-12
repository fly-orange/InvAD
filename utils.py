import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
import torch.nn.functional as F
import logging
from itertools import groupby
import sys
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from operator import itemgetter
from affiliation.generics import convert_vector_to_events
from affiliation.metrics import pr_from_events

class PyraMIL():
    def __init__(self):
        self.bce = nn.BCELoss(reduction='mean')

    def total_loss(self, scores, wlabel):
        depth = len(scores)
        loss=0
        
        for i in range(depth):
            wpred = torch.max(scores[i], dim=1)[0]

            loss += self.bce(wpred, wlabel)

        return loss/depth

    def weak_loss(self, scores, wlabel):
        
        wpred = scores[-1][:, 0]  # 只考虑粗标签
        # print(wpred.shape, wlabel.shape)
        loss = self.bce(wpred, wlabel)
        
        return loss

    def pyra_loss(self, scores, wlabel, depth):
        
        loss = 0
        return loss

    def last_loss(self, scores, wlabel):
        loss = self.bce(scores, wlabel)
        return loss


def ranking_loss(nscore, ascore):
    # nscore (B, T), ascore (B, T)
    maxn = torch.mean(nscore,dim=1)
    maxa = torch.mean(ascore,dim=1)
    tmp = F.relu(1. - maxa + maxn) # (B, )
    loss = tmp.mean()
    
    return loss

def compute_single_case(pred,label):
    
    correct = torch.mul(pred, label)
    TP, T, P = torch.sum(correct, dim=1), torch.sum(label, dim=1), torch.sum(pred,dim=1) # (B,)

    recall, precision, IoU = 1, 1, TP / (T + P - TP)  
    
    precision, recall, f1 = torch.ones(len(TP)), torch.ones(len(TP)), torch.ones(len(TP)) 
    for i in range(len(TP)):
        if T[i] != 0: recall[i] = TP[i] / T[i]   # (B)
        if P[i] != 0: precision[i] = TP[i]  / P[i]  # (B)

        if recall[i]==0.0 or precision[i]==0.0:
            f1[i] = 0.0
        else:
            f1[i] = 2*(recall[i]*precision[i])/(recall[i]+precision[i]) # (B)
    
    return f1, IoU


def compute_wacc(pred, label):
    correct = torch.mul(pred, label)
    TP, T, P = int(torch.sum(correct)), int(torch.sum(label)), int(torch.sum(pred))
    return np.array([TP, T, P])

def compute_dacc(pred, label):
    correct = torch.mul(pred, label)
    TP, T, P = int(torch.sum(correct)), int(torch.sum(label)), int(torch.sum(pred))
    return np.array([TP, T, P])

def compute_seen_and_unseen_recall(pred, label, kind):
    correct = torch.mul(pred, label) 
    seen_correct = correct[kind==1].sum()/(kind==1).sum()
    unseen_correct = correct[kind==2].sum()/(kind==2).sum()
    return seen_correct, unseen_correct

def compute_auc(pred, label):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().detach().numpy().flatten()
    if isinstance(label, torch.Tensor):
        label = label.cpu().detach().numpy().flatten()
    fpr, tpr, thresholds = metrics.roc_curve(label, pred)
    return metrics.auc(fpr, tpr)

def compute_bestf1(score, label, return_threshold=False):
    if isinstance(score, torch.Tensor):
        score = score.cpu().detach().numpy().flatten()
    if isinstance(label, torch.Tensor):
        label = label.cpu().detach().numpy().flatten()

    indices = np.argsort(score)[::-1]
    sorted_score = score[indices]
    sorted_label = label[indices]
    true_indices = np.where(sorted_label == 1)[0]

    bestf1 = 0.0
    best_threshold=None
    T = sum(label)
    for _TP, _P in enumerate(true_indices):
        TP, P = _TP + 1, _P + 1
        precision = TP / P
        recall = TP / T
        f1 = 2 * (precision*recall)/(precision+recall)
        threshold = sorted_score[_P] - np.finfo(float).eps
        if f1 > bestf1: # and threshold <= 0.5:
            bestf1 = f1
            best_threshold = sorted_score[_P] - np.finfo(float).eps
            #best_threshold = (sorted_score[_P-1] + sorted_score[_P]) / 2
    if return_threshold:
        return bestf1, best_threshold
    else:
        return bestf1

def compute_auprc(pred, label):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().detach().numpy().flatten()
    if isinstance(label, torch.Tensor):
        label = label.cpu().detach().numpy().flatten()
    return metrics.average_precision_score(label, pred)

def compute_precision_recall(result):
    TP, T, P = result
    recall, precision, IoU = 1, 1, TP / (T + P - TP)  
    if T != 0: recall = TP / T
    if P != 0: precision = TP / P

    if recall==0.0 or precision==0.0:
        f1 = 0.0
    else:
        f1 = 2*(recall*precision)/(recall+precision)
    return precision, recall, f1, IoU


'''Load params from the pretrained model'''

def load_model_checkpoint(checkpoint_file, model):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint)

    return model


def build_finetune_model(model_new, model_pretrained):
    """
    Load pretrained weights to Pyramid model
    """
    # Load in pre-trained parameters
    model_new.embedding = model_pretrained.embedding
    model_new.conv_layers = model_pretrained.conv_layers
    model_new.layers = model_pretrained.layers

    return model_new

def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # format_string = ("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
    #                 "%(lineno)d — %(message)s")
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger

def f1_prec_recall_PA(preds, gts, k=0):
    # Find out the indices of the anomalies
    if isinstance(preds, torch.Tensor):
            preds = preds.cpu().detach().numpy().flatten()
    if isinstance(gts, torch.Tensor):
        gts = gts.cpu().detach().numpy().flatten()

    gt_idx = np.where(gts == 1)[0]
    anomalies = []
    new_preds = np.array(preds)
    # Find the continuous index 
    for  _, g in groupby(enumerate(gt_idx), lambda x : x[0] - x[1]):
        anomalies.append(list(map(itemgetter(1), g)))
    # For each anomaly (point or seq) in the test dataset
    for a in anomalies:
        # Find the predictions where the anomaly falls
        pred_labels = new_preds[a]
        # Check how many anomalies have been predicted (ratio)
        if len(np.where(pred_labels == 1)[0]) / len(a) > (k/100):
            # Update the whole prediction range as correctly predicted
            new_preds[a] = 1
    f1_pa = f1_score(gts, new_preds)
    prec_pa = precision_score(gts, new_preds)
    recall_pa = recall_score(gts, new_preds)
    return f1_pa, prec_pa, recall_pa

def f1_prec_recall_K(preds, gts):

    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().detach().numpy().flatten()
    if isinstance(gts, torch.Tensor):
        gts = gts.cpu().detach().numpy().flatten()

    f1_pa_k = []
    prec_pa_k = []
    recall_pa_k = []
    for k in range(0,101):
        f1_pa, prec_pa, recall_pa = f1_prec_recall_PA(preds, gts, k)   
        f1_pa_k.append(f1_pa)
        prec_pa_k.append(prec_pa)
        recall_pa_k.append(recall_pa)
    f1_pak_auc = np.trapz(f1_pa_k)
    prec_pak_auc = np.trapz(prec_pa_k)
    recall_pak_auc = np.trapz(recall_pa_k)
    return f1_pak_auc/100, prec_pak_auc/100, recall_pak_auc/100
    

def aff_metrics(preds, gts):

    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().detach().numpy().flatten()
    if isinstance(gts, torch.Tensor):
        gts = gts.cpu().detach().numpy().flatten()

    events_pred = convert_vector_to_events(preds)
    if len(np.where(preds == 1)[0]) == 0 or len(np.where(gts == 1)[0]) == 0:
        print('all 0s no event to evaluate')
        return 0,0 
    events_gt = convert_vector_to_events(gts)
    Trange = (0, len(preds))
    aff_res = pr_from_events(events_pred, events_gt, Trange) 
    return aff_res['precision'], aff_res['recall']
