import argparse
import copy
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

# from loader_v2 import *
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from experiment_depth.depth_4_block_8 import ChannelEffFormer
from loader import *

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/work/scratch/arimond',
                    help='root dir for data')
parser.add_argument('--test_path', type=str,
                    default='./data/Synapse/test_vol_h5', help='root dir for data')
parser.add_argument('--weight', type=str, default='./results/ISIC17/swin_res34_cv_110_111_33_true_cfg.model',
                    help='dir for load weights for PH2 dataset')
parser.add_argument('--dataset', type=str,
                    default='ISIC17', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=160, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=1, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=1e-4,
                    help='segmentation network learning rate')
parser.add_argument('--num_workers', type=int,  default=2,
                    help='number of workers')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--output_dir', type=str,
                    default='./results', help='root dir for output log')
parser.add_argument('--model_name', type=str,
                    default='skin_det_block_8', help='[Swin_Res18, Swin_Res34, Swin_Res50]')
parser.add_argument('--eval_interval', type=int,
                    default=20, help='evaluation epoch')
parser.add_argument('--z_spacing', type=int,
                    default=1, help='z_spacing')

args = parser.parse_args()

args.output_dir = args.output_dir + f'/{args.dataset}/'
# args.save_name = args.output_dir + f'{args.model_name}.model'
args.save_name = args.output_dir + f'{args.model_name}'
os.makedirs(args.output_dir, exist_ok=True)


best_val_loss  = np.inf
device = 'cuda' if torch.cuda.is_available() else 'cpu'

##########
if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    print(dataset_name)
    dataset_config = {
        'ISIC17': {
            'root_path': os.path.join(args.root_path,"isic17_224/"),
            'num_classes': 1,
        },
        'ISIC18': {
            'root_path': os.path.join(args.root_path,"isic18_224/"),
            'num_classes': 1,
        },
        'PH2': {
            'root_path': os.path.join(args.root_path,"ph2_224/"),
            'num_classes': 1,
        }
    }

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24

    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    
    train_dataset = isic_loader(path_Data = args.root_path, train = True)
    train_loader  = DataLoader(train_dataset, batch_size = args.batch_size, shuffle= True)
    val_dataset   = isic_loader(path_Data = args.root_path, train = False, Test = False)
    val_loader    = DataLoader(val_dataset, batch_size = 5, shuffle= True)
    
    net = ChannelEffFormer(num_classes=args.num_classes).cuda()
    # if args.dataset == 'PH2': 
    #     net.load_state_dict(torch.load(args.weight, map_location="cuda:0")['model_weights'])
    #     print('weights loaded.')
    net = net.to(device)
        
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    optimizer = optim.Adam(net.parameters(), lr= args.base_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.5, patience = 10)
    criteria  = torch.nn.BCELoss()

    best_F1_score = 0.0
    for ep in range(args.max_epochs): # args.max_epochs
        net.train()
        epoch_loss = 0

        tqdm_loop = tqdm(enumerate(train_loader),
                         total=len(train_loader),
                         leave=False)

        for itter, batch in tqdm_loop:
            img = batch['image'].to(device, dtype=torch.float)
            msk = batch['mask'].to(device)
            mask_type = torch.float32
            msk = msk.to(device=device, dtype=mask_type)
            msk_pred = net(img)
            loss          = criteria(msk_pred, msk)
            optimizer.zero_grad()
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()

            # update progress bar
            tqdm_loop.set_description(f"Epoch [{ep + 1}/{args.max_epochs}]")
            tqdm_loop.set_postfix(loss=loss.item())

        predictions = []
        gt = []

        if (ep+1)%args.eval_interval==0:
            with torch.no_grad():
                print('val_mode')
                val_loss = 0
                net.eval()
                for itter, batch in enumerate(val_loader):
                    img = batch['image'].to(device, dtype=torch.float)
                    msk = batch['mask']
                    msk_pred = net(img)
                    gt.append(msk.numpy()[0, 0])
                    msk_pred = msk_pred.cpu().detach().numpy()[0, 0]
                    msk_pred  = np.where(msk_pred>=0.5, 1, 0)

                    predictions.append(msk_pred)        

            predictions = np.array(predictions)
            gt = np.array(gt)

            y_scores = predictions.reshape(-1)
            y_true   = gt.reshape(-1)

            y_scores2 = np.where(y_scores>0.5, 1, 0)
            y_true2   = np.where(y_true>0.5, 1, 0)

            #F1 score
            F1_score = f1_score(y_true2, y_scores2, labels=None, average='binary', sample_weight=None)
            print (f"\nF1 score (F-measure) or DSC in epoch {ep+1}:  {F1_score}")
            if (F1_score) > best_F1_score:
                print('New best loss, saving...')
                best_F1_score = copy.deepcopy(F1_score)
                state = copy.deepcopy({'model_weights': net.state_dict(), 'test_F1_score': F1_score})
                torch.save(state, args.save_name+".model")
                print(args.save_name+".model")
#             if (F1_score) > best_F1_score or F1_score >= 0.96:
#                 print('New best loss, saving...')
#                 best_F1_score = copy.deepcopy(F1_score)
#                 state = copy.deepcopy({'model_weights': net.state_dict(), 'test_F1_score': F1_score})
#                 torch.save(state, args.save_name + f"_{F1_score}_ep_{ep}.model")
#                 print(args.save_name + f"_{F1_score}_ep_{ep}.model")
                
                
            