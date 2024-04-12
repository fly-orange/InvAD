import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datetime import datetime
from utils import _logger
import os, time
import argparse
import copy
from dataloader.timeseries import TimeSeriesWithAnomalies, TimeSeriesWithAnomalies2, NormalTimeSeries, AbnormalTimeSeries
from utils import * 


# Arguments parsing
parser = argparse.ArgumentParser()
# for Dataset
parser.add_argument('--dataset', default='EMG', type=str, help='EMG | GHL | SMD | SMAP | PSM | MSL')
# for Model
parser.add_argument('--method', default='scnet', type=str, help='Model type')
# for Pretrain
parser.add_argument('--test', default=False, action='store_true', help='train or test')
parser.add_argument('--fine_tune', default=False, action='store_true', help='whether to fine tune the pretrained model')

# for settings
parser.add_argument('--device', default='cuda', type=str, help='cpu or cuda')
parser.add_argument('--gpuidx', default=1, type=int, help='gpu index')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str, help='saving directory')
# parser.add_argument('--experiment_description', default='Exp1', type=str, help='Experiment Description')
# parser.add_argument('--run_description', default='run1', type=str, help='Experiment Description')


args = parser.parse_args()

# GPU setting
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuidx)
os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)

# Random seed initialization
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
import random
random.seed(args.seed)

args.data_dir = './data/' + args.dataset
method = args.method.upper()
device = torch.device(args.device)
print(args.data_dir)

exec(f'from conf.{args.method}.{args.dataset}_Configs import Config as Configs')
exec(f'from models.{method}.{args.method}_network.model import base_Model')
if args.dataset in ['EPI', 'RS', 'NATOPS', 'CT', 'SAD']:
    exec(f'from models.{method}.{args.method}_trainer.wtrainer import Trainer')
    exec(f'from models.{method}.{args.method}_trainer.wtrainer import Tester')
else:
    exec(f'from models.{method}.{args.method}_trainer.trainer import Trainer')
    exec(f'from models.{method}.{args.method}_trainer.trainer import Tester')


configs = Configs()

experiment_log_dir = os.path.join(args.logs_save_dir, args.dataset, f"_seed_{args.seed}")
os.makedirs(experiment_log_dir, exist_ok=True)
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Dataset: {args.dataset}')
logger.debug(f'Method:  {method}')
logger.debug("=" * 45)


'''Load Data'''

if args.dataset in ['EPI', 'RS', 'NATOPS', 'CT', 'SAD']:
    train_dataset = TimeSeriesWithAnomalies2(args.data_dir, 'train', configs)
    valid_dataset = TimeSeriesWithAnomalies2(args.data_dir, 'valid', configs)
    test_dataset = TimeSeriesWithAnomalies2(args.data_dir, 'test', configs)
else:
    train_dataset = TimeSeriesWithAnomalies(args.data_dir, configs.window_size, 'train', configs)
    valid_dataset = TimeSeriesWithAnomalies(args.data_dir, configs.window_size, 'valid', configs)
    test_dataset = TimeSeriesWithAnomalies(args.data_dir, configs.window_size, 'test', configs)
train_loader = DataLoader(dataset=train_dataset, batch_size=configs.batch_size, shuffle=True, num_workers=1)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=configs.batch_size, shuffle=False, num_workers=1)
test_loader = DataLoader(dataset=test_dataset, batch_size=configs.batch_size, shuffle=False, num_workers=1)
logger.debug("Data Loaded")


'''Load Model'''
# for winsize in range(120,481,120):
#     configs.window_size = winsize
# configs.lambda2=0
# logger.debug(f"No contrastive")
model = base_Model(configs, device).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2),
                                # weight_decay=configs.weight_decay)
optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)
logger.debug("Model Loaded")

'''Training'''
if not args.test:
    wauc = Trainer(args, model, optimizer, train_loader, valid_loader, test_loader, device, logger, configs)
    logger.debug(f"wacuc: {wauc}")
elif args.test:
    Tester(args, model, train_loader, valid_loader, test_loader, device, logger, configs)


# configs.lambda2=1
# for sup in ['sup','unsup']:
#     configs.learning_mode=sup 
#     logger.debug(f"Contrative strategy: {configs.learning_mode}")
#     model = base_Model(configs, device).to(device)
#     # optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2),
#                                     # weight_decay=configs.weight_decay)
#     optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)
#     logger.debug("Model Loaded")

#     '''Training'''
#     if not args.test:
#         wauc, dauc = Trainer(args, model, optimizer, train_loader, valid_loader, test_loader, device, logger, configs)
#         logger.debug(f"wacuc: {wauc}, dacuc: {dauc}")
#     elif args.test:
#         Tester(args, model, train_loader, valid_loader, test_loader, device, logger, configs)