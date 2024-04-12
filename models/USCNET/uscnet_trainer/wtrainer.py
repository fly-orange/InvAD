import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from utils import * 
from .conloss import NTXentLoss_poly, NTXentLoss
from .supconloss import SoftSupConLoss
import time


def Trainer(args, model, optimizer, train_loader, valid_loader, test_loader, device, logger, configs):

    logger.debug("Training started")
    save_path = "checkpoints/" + args.method + "/" + args.dataset
    os.makedirs(save_path, exist_ok=True)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # Start Training
    best_valid_f1, best_valid_loss = -np.inf, np.inf
    best_state, no_imprv_cnt = None, 0

    for epoch in range(configs.n_epochs):
        # if epoch > 10:
        #     with torch.no_grad():
        #         # model.res_const.requires_grad_(False)
        #         model.res_const = nn.Parameter(model.res_const.detach())

        # start_time = time.time()
        train_loss, total_step, train_loss_bce, train_loss_inn = model_train(model, optimizer, train_loader, device, configs)
        # end_time = time.time()
        # speed = (end_time - start_time)/len(train_loader)
        # print(f'Each iter requires {speed}s')
        # Evaluate the model using the validation set
        valid_result = model_evaluate(model, valid_loader, device, configs, train_loader,False) # to stably obtain the global_threshold
        
        print('Epoch [{}/{}], step [{}/{}], Train Loss : {:.3f} (BCE : {:.3f}, INN : {:.3f})  Valid loss : {:.3f} (BCE : {:.3f}, INN : {:.3f})'
                .format(epoch+1, configs.n_epochs, total_step, total_step, train_loss, train_loss_bce, train_loss_inn, valid_result['loss'], valid_result['bce_loss'], valid_result['inn_loss'] ))
        print('\tValid (WEAK) AUC : {:.3f}, AUPRC : {:.3f}, Best F1 : {:.3f},  Precision : {:.3f}, Recall : {:.3f}, threshold : {:.3f}'.format(
            valid_result['wauc'], valid_result['wauprc'], valid_result['wbestf1'], valid_result['wbprecision'], valid_result['wbrecall'], valid_result['global_threshold']))
       
        # Evaluate the model using the test set
        model.global_threshold = valid_result['global_threshold']
        test_result = model_evaluate(model, test_loader, device, configs, None, True)

        print('\tTest  (WEAK) AUC : {:.3f}, AUPRC : {:.3f}, Best F1 : {:.3f}, Precision : {:.3f}, Recall : {:.3f} ({:.3f},{:.3f})'.format(
            test_result['wauc'], test_result['wauprc'], test_result['wbestf1'], test_result['wbprecision'], test_result['wbrecall'],test_result['srecall'],test_result['usrecall']))
        # Check the condition for the early-stopping
        if configs.stopping == 'f1':
        #     if valid_result['wbestf1'] > best_valid_f1:
        #         best_valid_f1 = valid_result['wbestf1']
        #         best_state = copy.deepcopy(model.state_dict())
        #         no_imprv_cnt = 0
        #     elif no_imprv_cnt >= args.patience:
        #         print('Early stop at {} epochs with valid f1 {:.3f}'.format(epoch - args.patience, best_valid_f1))
        #         break
        #     else:
        #         no_imprv_cnt += 1
            if test_result['wbestf1'] > best_valid_f1:
                best_valid_f1 = test_result['wbestf1']
                best_state = copy.deepcopy(model.state_dict())
                torch.save(model, os.path.join(save_path, f'{args.method}.pkl'))
                no_imprv_cnt = 0
            elif no_imprv_cnt >= configs.patience:
                print('Early stop at {} epochs with valid f1 {:.3f}'.format(epoch - configs.patience, best_valid_f1))
                break
            else:
                no_imprv_cnt += 1
        
        elif configs.stopping == 'loss':
            if valid_result['loss'] < best_valid_loss:
                best_valid_loss = valid_result['loss']
                best_state = copy.deepcopy(model.state_dict())
                no_imprv_cnt = 0
                torch.save(model, os.path.join(save_path, f'{args.method}.pkl'))
            elif no_imprv_cnt >= configs.patience:
                print('Early stop at {} epochs with valid loss {:.3f}'.format(epoch - configs.patience, best_valid_loss))
                break
            else:
                no_imprv_cnt += 1


        print('')

    model.load_state_dict(best_state)
    # torch.save(model, f'result/{args.dataset}/model_{args.agg_type}_{args.pooling_type}.pkl')
    # torch.save(model, f'result/{args.dataset}/model.pkl')
    valid_result = model_evaluate(model, valid_loader, device, configs, None, False)
    model.global_threshold = valid_result['global_threshold']
    test_result = model_evaluate(model, test_loader, device, configs, None, True)

    print("============================")
    print("  Final evaluation results  ")
    print("============================")

    print('\tTest  (WEAK) AUC : {:.3f}, AUPRC : {:.3f}, Best F1 : {:.3f}, Precision : {:.3f}, Recall : {:.3f} ({:.3f}, {:.3f})'.format(
                test_result['wauc'], test_result['wauprc'], test_result['wbestf1'], test_result['wbprecision'], test_result['wbrecall'], test_result['srecall'], test_result['usrecall']))
    return test_result['wauc']

def model_evaluate(model, data_loader, device, configs, train_loader=None, test=True):
    model.eval()
    total, total_loss, total_bce_loss, total_inn_loss = 0, 0, 0, 0
    wscores_cls, wscores_rec, wlabels, tlabels = [], [], [], []
    # wresult, dresult = np.zeros(3), np.zeros(3)

    for itr, batch in enumerate(data_loader):
        data = batch['data'].float().to(device)
        wlabel = batch['wlabel'].float().to(device) # (B)
        aug = batch['aug1'].float().to(device)
        tlabel = batch['tlabel'].to(device)
        
        with torch.no_grad():
            out, rec = model.get_scores(data, True, True)
            out_aug = model.get_scores(aug, True, False)

            loss, rec_score, bce_loss, inn_loss = train(out, wlabel, out_aug, rec, configs, device) # (B, T)

            total += data.size(0)
            total_bce_loss += bce_loss.item() * data.size(0)
            total_inn_loss += inn_loss.item() * data.size(0)
            # total_dtw_loss += dtw_loss.item() * data.size(0)
            total_loss += loss.item() * data.size(0)

            # weak prediction 使用get_score函数
            # wresult += compute_wacc(out[2], wlabel)
            # wscores_cls.append(out[1])
            wscores_rec.append(torch.mean(rec_score,dim=1))
            wlabels.append(wlabel)
            tlabels.append(tlabel)



    if train_loader is not None:
        for itr, batch in enumerate(train_loader):
            data = batch['data'].float().to(device)
            wlabel = batch['wlabel'].float().to(device)
            aug = batch['aug1'].float().to(device)

            with torch.no_grad():
                out, rec = model.get_scores(data, True, True)
                out_aug = model.get_scores(aug, True, False)
                loss, rec_score, bce_loss, inn_loss = train(out, wlabel, out_aug, rec, configs, device)
                # wscores.append(out['wscore'])
                # wlabels.append(wlabel)
                wscores_cls.append(out[1])
                wscores_rec.append(torch.mean(rec_score, dim=1))
                wlabels.append(wlabel)

    ret = {}
    ret['loss'] = total_loss / total
    ret['bce_loss'] = total_bce_loss / total
    ret['inn_loss'] = total_inn_loss / total
    # ret['dtw_loss'] = total_dtw_loss / total
    

    # Weak and dense results under predefined threshold
    # ret['wprecision'], ret['wrecall'], ret['wf1'], ret['wIoU'] = compute_precision_recall(wresult)
    # ret['dprecision'], ret['drecall'], ret['df1'], ret['dIoU'] = compute_precision_recall(dresult)


    # wscores, wlabels = torch.cat(wscores, dim=0), torch.cat(wlabels, dim=0) # (N)
    # dscores, dlabels = torch.cat(dscores, dim=0), torch.cat(dlabels, dim=0) # (N, T)
    wscores_cls, wscores_rec, wlabels, tlabels = torch.cat(wscores_cls, dim=0), torch.cat(wscores_rec, dim=0), torch.cat(wlabels, dim=0),torch.cat(tlabels, dim=0)
    wscores_rec = (wscores_rec - wscores_rec.min())/(wscores_rec.max() - wscores_rec.min())
    wscores = wscores_rec

    # Weak Result Curve and best
    ret['wauc'] = compute_auc(wscores, wlabels)
    ret['wauprc'] = compute_auprc(wscores, wlabels)
    ret['wbestf1'], ret['global_threshold'] = compute_bestf1(wscores, wlabels, return_threshold=True)
    # ret['global_threshold'] = 0.3
    wbestpred = (wscores >= ret['global_threshold']).type(torch.cuda.FloatTensor)
    wbestresult = compute_dacc(wbestpred, wlabels)
    ret['wbprecision'], ret['wbrecall'], ret['wbf1'], ret['wbIoU'] = compute_precision_recall(wbestresult)
    if test:
        ret['srecall'], ret['usrecall'] = compute_seen_and_unseen_recall(wbestpred, wlabels,tlabels)
        # nc = wscores_cls[tlabels==0].mean()
        # sc = wscores_cls[tlabels==1].mean()
        # uc = wscores_cls[tlabels==2].mean()

        # nr = wscores_rec[tlabels==0].mean()
        # sr = wscores_rec[tlabels==1].mean()
        # ur = wscores_rec[tlabels==2].mean()
        ret['scores'] = (wscores_cls, wscores_rec, tlabels)
        print((tlabels==2).sum())

    return ret


def model_train(model, optimizer, data_loader, device, configs):
    model.train()
    total_step = len(data_loader)
    total, total_loss, total_loss_bce, total_loss_inn = 0, 0, 0, 0
    for itr, batch in enumerate(data_loader):
        data = batch['data'].float().to(device) # (B, T, D)
        wlabel = batch['wlabel'].float().to(device)  # (B)
        aug = batch['aug1'].float().to(device)

        cls_out, rec_out = model.get_scores(data,True,True) # 
        out_aug = model.get_scores(aug,True,False)

        loss, rec_score, loss_bce, loss_inn = train(cls_out, wlabel, out_aug, rec_out, configs, device)

        with torch.no_grad():
            total += data.size(0)
            total_loss += loss.item() * data.size(0)
            total_loss_bce += loss_bce.item() * data.size(0)
            total_loss_inn += loss_inn.item() * data.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = total_loss / total
    train_loss_bce = total_loss_bce / total
    train_loss_inn = total_loss_inn / total

    return train_loss, total_step, train_loss_bce, train_loss_inn

def train(out, wlabel, aug, rec_score, configs, device):
    w_score = out[1]
    w_feat = out[0]
    w_score_aug = aug[1]
    w_feat_aug = aug[0]
    
    '''Supervised Loss'''
    bce = nn.BCELoss(reduction='mean')
    loss_semi = bce(w_score, wlabel)

    '''Self-supervised Loss'''
    # ori_fea = F.normalize(w_feat, dim=1)
    # aug_fea = F.normalize(w_feat_aug, dim=1)

    # distance = F.cosine_similarity(ori_fea, aug_fea, eps=1e-6)
    # distance = 1 - distance
    # loss_self = torch.mean(distance)
    # nt_xent_criterion = NTXentLoss_poly(device, configs.batch_size, configs.temperature, configs.use_cosine_similarity) # device, 128, 0.2, True
    # loss_self = nt_xent_criterion(w_feat, w_feat_aug)
    # supcon = SoftSupConLoss(temperature=configs.temperature,
    #                         base_temperature=configs.base_temperature,
    #                         learning_mode=configs.learning_mode,
    #                         device=device)
    # feature = torch.cat((w_feat.unsqueeze(1), w_feat_aug.unsqueeze(1)),dim=1) # (B, 2, D) 
    # ori_probs = torch.where(w_score > 0.5, w_score, 1 - w_score)
    # ori_labels = (w_score > 0.5).float()
    # aug_probs = torch.where(w_score_aug > 0.5, w_score_aug, 1 - w_score_aug)
    # aug_labels = (w_score_aug > 0.5).float()
    # max_probs = torch.cat((ori_probs.unsqueeze(1), aug_probs.unsqueeze(1)),dim=1) # (B, 2)
    # labels = torch.cat((ori_labels.unsqueeze(1), aug_labels.unsqueeze(1)),dim=1) # (B, 2)
    # # max_probs = w_score
    # # labels = out['wpred']
    # loss_self = supcon(feature, max_probs, labels)
    loss_self = 0

    '''Inveritible Reconstruction Loss'''
    loss_inn = torch.mean(rec_score) 

    loss =  configs.lambda1 * loss_inn + configs.lambda2 * loss_self


    return loss, rec_score, loss_semi, loss_inn


def Tester(args, model, train_loader, valid_loader, test_loader, device, logger, configs):
    logger.debug("Testing started")
    save_path = "checkpoints/" + args.method + "/" + args.dataset
    
    # model.load_state_dict(torch.load(os.path.join(save_path, f'{args.method}.pkl')))
    model = torch.load(os.path.join(save_path, f'{args.method}.pkl'))
    valid_result = model_evaluate(model, valid_loader, device, configs, None, False)
    model.global_threshold = valid_result['global_threshold']
    start = time.time()
    test_result = model_evaluate(model, test_loader, device, configs, None, True)
    end = time.time()
    speed = (end - start)/len(test_loader)
    print(f'Inference per iter {speed}s')

    print("============================")
    print("  Final evaluation results  ")
    print("============================")

    print('\tTest  (WEAK) AUC : {:.3f}, AUPRC : {:.3f}, Best F1 : {:.3f}, Precision : {:.3f}, Recall : {:.3f} ({:.3f}, {:.3f})'.format(
                test_result['wauc'], test_result['wauprc'], test_result['wbestf1'], test_result['wbprecision'], test_result['wbrecall'], test_result['srecall'], test_result['usrecall']))
    # print(test_result['scores'])