import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from utils import * 


def Trainer(args, model, train_loader, valid_loader, test_loader, device, logger, configs):
    logger.debug("Testing started")
    save_path = "checkpoints/" + args.method + "/" + args.dataset
    
    # model.load_state_dict(torch.load(os.path.join(save_path, f'{args.method}.pkl')))
    # model = torch.load(os.path.join(save_path, f'{args.method}.pkl'))
    valid_result = model_test(model, valid_loader, device, configs, None)
    model.global_threshold = valid_result['global_threshold']
    test_result = model_test(model, test_loader, device, configs, None)

    print("============================")
    print("  Final evaluation results  ")
    print("============================")

    print('\tTest  (WEAK) AUC : {:.3f}, AUPRC : {:.3f}, Best F1 : {:.3f}, Precision : {:.3f}, Recall : {:.3f}'.format(
                test_result['wauc'], test_result['wauprc'], test_result['wbestf1'], test_result['wbprecision'], test_result['wbrecall']))
    print('\tTest (DENSE) AUC : {:.3f}, AUPRC : {:.3f}, F1 : {:.3f}, Precision : {:.3f}, Recall : {:.3f}, IoU : {:.3f}'.format(
        test_result['dauc'], test_result['dauprc'], test_result['dbf1'], test_result['dbprecision'], test_result['dbrecall'], test_result['dbIoU']))
    print('\tTest F1_PA: {:.3f}, F1_PAK_AUC: {:.3f}, Pre_PAK_AUC: {:.3f}, Recall_PAK_AUC: {:.3f}, F1_Aff: {:.3f}, Pre_Aff: {:.3f}, Recall_Aff: {:.3f}'.format(test_result['f1_pa'], test_result['f1_pak_auc'], test_result['prec_pak_auc'], test_result['recall_pak_auc'], test_result['f1_aff'], test_result['prec_aff'], test_result['recall_aff']))


def model_test(model, data_loader, device, configs, train_loader=None):
    model.eval()
    wlabels, wscores, dlabels, dscores = [], [], [], []

    for itr, batch in enumerate(data_loader):
        data = batch['data'].to(device)
        wlabel = batch['wlabel'].to(device) # (B)
        dlabel = batch['dlabel'].to(device) # (B, T)
        
        with torch.no_grad():
            wscore, dscore = model.get_scores(data)

            wlabels.append(wlabel)
            wscores.append(wscore)
            dscores.append(dscore)
            dlabels.append(dlabel)

    if train_loader is not None:
        for itr, batch in enumerate(train_loader):
            data = batch['data'].to(device)
            wlabel = batch['wlabel'].to(device)

            with torch.no_grad():

                wscore, dscore = model.get_scores(data)

                wlabels.append(wlabel)
                wscores.append(wscore)
                dscores.append(dscore)
                dlabels.append(dlabel)

    ret = {}

    # wscores, wlabels = torch.cat(wscores, dim=0), torch.cat(wlabels, dim=0) # (N)
    # dscores, dlabels = torch.cat(dscores, dim=0), torch.cat(dlabels, dim=0) # (N, T)
    wscores, wlabels = torch.cat(wscores, dim=0),  torch.cat(wlabels, dim=0)
    dscores, dlabels = torch.cat(dscores, dim=0),  torch.cat(dlabels, dim=0)
    wscores = (wscores - wscores.min())/(wscores.max() - wscores.min())
    dscores = (dscores - dscores.min())/(dscores.max() - dscores.min())

    # Weak Result Curve and best
    ret['wauc'] = compute_auc(wscores, wlabels)
    ret['wauprc'] = compute_auprc(wscores, wlabels)
    ret['wbestf1'], ret['global_threshold'] = compute_bestf1(wscores, wlabels, return_threshold=True)
    wbestpred = (wscores >= ret['global_threshold']).type(torch.cuda.FloatTensor)
    wbestresult = compute_dacc(wbestpred, wlabels)
    ret['wbprecision'], ret['wbrecall'], ret['wbf1'], ret['wbIoU'] = compute_precision_recall(wbestresult)


    # Dense Result Curve and best
    ret['dauc'] = compute_auc(dscores, dlabels)
    ret['dauprc'] = compute_auprc(dscores, dlabels)
    ret['dbestf1'], ret['local_threshold'] = compute_bestf1(dscores, dlabels, return_threshold=True)
    dbestpred = (dscores >= ret['local_threshold']).type(torch.cuda.FloatTensor)
    dbestresult = compute_dacc(dbestpred, dlabels)
    ret['dbprecision'], ret['dbrecall'], ret['dbf1'], ret['dbIoU'] = compute_precision_recall(dbestresult)

    ret['f1_pa'],_,_ = f1_prec_recall_PA(dbestpred, dlabels)
    ret['f1_pw'],_,_ = f1_prec_recall_PA(dbestpred, dlabels, k=100)
    ret['f1_pak_auc'], ret['prec_pak_auc'], ret['recall_pak_auc'] = f1_prec_recall_K(dbestpred, dlabels)
    ret['prec_aff'], ret['recall_aff'] = aff_metrics(dbestpred, dlabels)
    ret['f1_aff'] = 2 * (ret['prec_aff'] * ret['recall_aff']) / (ret['prec_aff'] + ret['recall_aff'])

    return ret



def Tester(args, model, train_loader, valid_loader, test_loader, device, logger, configs):
    logger.debug("Testing started")
    save_path = "checkpoints/" + args.method + "/" + args.dataset
    
    # model.load_state_dict(torch.load(os.path.join(save_path, f'{args.method}.pkl')))
    # model = torch.load(os.path.join(save_path, f'{args.method}.pkl'))
    valid_result = model_test(model, valid_loader, device, configs, None)
    model.global_threshold = valid_result['global_threshold']
    test_result = model_test(model, test_loader, device, configs, None)

    print("============================")
    print("  Final evaluation results  ")
    print("============================")

    print('\tTest  (WEAK) AUC : {:.3f}, AUPRC : {:.3f}, Best F1 : {:.3f}, Precision : {:.3f}, Recall : {:.3f}'.format(
                test_result['wauc'], test_result['wauprc'], test_result['wbestf1'], test_result['wbprecision'], test_result['wbrecall']))
    print('\tTest (DENSE) AUC : {:.3f}, AUPRC : {:.3f}, F1 : {:.3f}, Precision : {:.3f}, Recall : {:.3f}, IoU : {:.3f}'.format(
        test_result['dauc'], test_result['dauprc'], test_result['dbf1'], test_result['dbprecision'], test_result['dbrecall'], test_result['dbIoU']))
    print('\tTest F1_PA: {:.3f}, Test F1_PW: {:.3f}, F1_PAK_AUC: {:.3f}, Pre_PAK_AUC: {:.3f}, Recall_PAK_AUC: {:.3f}, F1_Aff: {:.3f}, Pre_Aff: {:.3f}, Recall_Aff: {:.3f}'.format(test_result['f1_pa'], test_result['f1_pw'], test_result['f1_pak_auc'], test_result['prec_pak_auc'], test_result['recall_pak_auc'], test_result['f1_aff'], test_result['prec_aff'], test_result['recall_aff']))