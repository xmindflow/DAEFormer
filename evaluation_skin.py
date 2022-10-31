import argparse
import os
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from loader import *
# from loader_v2 import *
from sklearn.metrics import f1_score
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import copy 
import models.SwinRetina_configs as configs
from models.SwinRetina_App3 import SwinRetina_V3
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy.ndimage.morphology import binary_dilation, binary_erosion, binary_fill_holes, binary_opening, binary_closing

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/staff/azad/deeplearning/transformer/yuli_version/processed_data/',
                    help='root dir for data')
parser.add_argument('--test_path', type=str,
                    default='./data/Synapse/test_vol_h5', help='root dir for data')
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
                    default=24, help='batch_size per gpu')
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
                    default='swin_res34_cv_11_11_612_true_cfg', help='[Swin_Res18, Swin_Res34, Swin_Res50]')
parser.add_argument('--eval_interval', type=int,
                    default=20, help='evaluation epoch')
parser.add_argument('--z_spacing', type=int,
                    default=1, help='z_spacing')

args = parser.parse_args()


best_val_loss  = np.inf
device = 'cuda' if torch.cuda.is_available() else 'cpu'

CONFIGS = {
        'swin_res34_cv_11_11_612_true_cfg'  : configs.get_swin_res34_cv_11_11_612_true_cfg(),
        'swin_res34_cv_120_221_612_true_cfg': configs.get_swin_res34_cv_120_221_612_true_cfg(),
        'swin_res34_cv_120_111_612_true_cfg': configs.get_swin_res34_cv_120_111_612_true_cfg(),
        'swin_res34_cv_120_221_66_true_cfg' : configs.get_swin_res34_cv_120_221_66_true_cfg(),
        'swin_res34_cv_11_11_44_true_cfg'   : configs.get_swin_res34_cv_11_11_44_true_cfg(),
        'swin_res34_cv_11_11_33_true_cfg'   : configs.get_swin_res34_cv_11_11_33_true_cfg(),
        'swin_res34_cv_110_111_612_true_cfg': configs.get_swin_res34_cv_110_111_612_true_cfg(),
        'swin_res34_cv_130_331_66_true_cfg' : configs.get_swin_res34_cv_130_331_66_true_cfg(),
        'swin_res34_cv_110_221_612_true_cfg': configs.get_swin_res34_cv_110_221_612_true_cfg(),
        'swin_res34_cv_140_441_66_true_cfg' : configs.get_swin_res34_cv_140_441_66_true_cfg(),
        'swin_res50_cv_110_111_612_true_cfg': configs.get_swin_res50_cv_110_111_612_true_cfg(),
        'swin_res50_cv_120_221_612_true_cfg': configs.get_swin_res50_cv_120_221_612_true_cfg(),
        'swin_res50_cv_120_221_66_true_cfg' : configs.get_swin_res50_cv_120_221_66_true_cfg(),
        'swin_res50_cv_130_331_66_true_cfg' : configs.get_swin_res50_cv_130_331_66_true_cfg(),
        'swin_res50_cv_120_111_612_true_cfg': configs.get_swin_res50_cv_120_111_612_true_cfg(),
        'swin_res50_cv_120_111_66_true_cfg' : configs.get_swin_res50_cv_120_111_66_true_cfg(),
        'swin_res50_cv_11_11_612_true_cfg'  : configs.get_swin_res50_cv_11_11_612_true_cfg(),
        'swin_res18_cv_11_11_612_true_cfg'  : configs.get_swin_res18_cv_11_11_612_true_cfg(),
        'swin_res34_cv_110_111_33_true_cfg' : configs.get_swin_res34_cv_110_111_33_true_cfg(),
        'swin_res50_cv_110_111_33_true_cfg' : configs.get_swin_res50_cv_110_111_33_true_cfg(),
        'swin_res50_cv_11_11_33_true_cfg'   : configs.get_swin_res50_cv_11_11_33_true_cfg(),
    }

args.output_dir = args.output_dir + f'/{args.dataset}/'
# args.output_dir = args.output_dir + '/isic18/'
args.save_name = args.output_dir + f'{args.model_name}.model'
# args.save_name = args.output_dir + 'swin_res50_cv_120_221_66_true_cfg_0.9224440483212881_ep_123.model'
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
            'root_path': args.root_path + "isic17_224/",
            'num_classes': 1,
        },
        'ISIC18': {
            'root_path': args.root_path + "isic18_224/",
            'num_classes': 1,
        },
        'PH2': {
            'root_path': args.root_path + "ph2_224/",
            'num_classes': 1,
        }
    }

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24

    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    
    
    test_dataset = isic_loader(path_Data = args.root_path, train = False, Test = True)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle= True)
    
    net = SwinRetina_V3(config=CONFIGS[args.model_name], img_size=args.img_size, n_classes=args.num_classes).cuda()
    net = net.to(device)
    
    optimizer = optim.Adam(net.parameters(), lr= args.base_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.5, patience = 10)
    criteria  = torch.nn.BCELoss()
    
    ## Load the weight file
    net.load_state_dict(torch.load(args.save_name, map_location='cpu')['model_weights'])
    print(f'model leaded from: {args.save_name}')
       
    predictions = []
    gt = []

    with torch.no_grad():
        print('val_mode')
        print('\n', args.save_name)
        val_loss = 0
        net.eval()

        tqdm_loop = tqdm(enumerate(test_loader),
                         total=len(test_loader),
                         leave=False,)

        for itter, batch in tqdm_loop:
            img = batch['image'].to(device, dtype=torch.float)
            msk = batch['mask']
            msk_pred = net(img)

            gt.append(msk.numpy()[0, 0])
            msk_pred = msk_pred.cpu().detach().numpy()[0, 0]
            msk_pred  = np.where(msk_pred>=0.9, 1, 0) ## we can try some Threshold values here
            ## We can add morphological operations here
            j = 3
#                     msk_pred = binary_dilation(msk_pred, structure=np.ones((j+1,j+1))).astype(msk_pred.dtype)
#                     msk_pred = binary_fill_holes(msk_pred, structure=np.ones((j+1,j+1))).astype(msk_pred.dtype)
#                     msk_pred = binary_erosion(msk_pred, structure=np.ones((j+1,j+1))).astype(msk_pred.dtype)
#             msk_pred = binary_opening(msk_pred, structure=np.ones((j+1,j+1))).astype(msk_pred.dtype)
            #msk_pred = binary_closing(msk_pred, structure=np.ones((j+1,j+1))).astype(msk_pred.dtype)
#             msk_pred = binary_fill_holes(msk_pred, structure=np.ones((j+1,j+1))).astype(msk_pred.dtype)



            predictions.append(msk_pred)

            # update progress bar
            tqdm_loop.set_description(f"Predicting: ")


    predictions = np.array(predictions)
    gt = np.array(gt)

    y_scores = predictions.reshape(-1)
    y_true   = gt.reshape(-1)

    y_scores2 = np.where(y_scores>0.5, 1, 0)
    y_true2   = np.where(y_true>0.6, 1, 0)

    #F1 score
    F1_score = f1_score(y_true2, y_scores2, labels=None, average='binary', sample_weight=None)

    print ("\nF1 score (F-measure) or DSC: " +str(F1_score))
#             dsc.append(F1_score)
    confusion = confusion_matrix(np.int32(y_true2), y_scores2)
    print (confusion)
    accuracy = 0
    if float(np.sum(confusion))!=0:
        accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
    print ("Accuracy: " +str(accuracy))
#             acc.append(accuracy)
    specificity = 0
    if float(confusion[0,0]+confusion[0,1])!=0:
        specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
    print ("Specificity: " +str(specificity))
#             sp.append(specificity)
    sensitivity = 0
    if float(confusion[1,1]+confusion[1,0])!=0:
        sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
    print ("Sensitivity: " +str(sensitivity))
#             se.append(sensitivity)

#             print("\n")
#             print(f'Morphology Kernel is {mk}, Threshold is {thresh} for model: {args.model_name}.') 
#             print("\n\n")
    
#     dict = {'Threshold': th, 'Kernel_Size': ker_sz, 'Dice': dsc, 'Accuracy': acc, 'Sensitivity': se, 'Specificity': sp} 
#     df = pd.DataFrame(dict)
#     df.to_csv(f'./enhanced_results_{args.model_name}_{args.dataset}.csv',index=False)
#     print('\n')
#     print(f'Now EXPLORE THE [ enhanced_results_{args.model_name}_{args.dataset}.csv ]')
#     print('\n \n')