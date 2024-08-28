import time

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from data.triplet_sampler import *
from loss.losses import triplet_loss_fastreid
from lr_scheduler.sche_optim import make_optimizer, make_warmup_scheduler
import argparse
import torch.multiprocessing
import yaml
import os
from tensorboard_log import Logger
from processor import get_model, train_epoch, test_epoch
from fvcore.nn import FlopCountAnalysis, parameter_count_table

torch.multiprocessing.set_sharing_strategy('file_system')

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['NCCL_P2P_DISABLE'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES']="-1"

def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic= True
    torch.backends.cudnn.benchmark= False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ReID model trainer')

    ############### Config #########################
    parser.add_argument('--config', default=None, help='Config Path')


    ############### DataSet  #######################
    parser.add_argument('--batch_size', default=None, type=int, help='Batch size')
    parser.add_argument('--hflip', default=None, type=float, help='Probabilty for horizontal flip')
    parser.add_argument('--randomerase', default=None, type=float,  help='Probabilty for random erasing')
    parser.add_argument('--dataset', default=None, help='Choose one of [Veri776, VERIWILD, Market1501, VehicleID]')
    parser.add_argument('--imgsize_x', default=None, type=int, help='width image')
    parser.add_argument('--imgsize_y', default=None, type=int, help='height image')
    parser.add_argument('--num_instances', default=None, type=int, help='Number of images belonging to an ID inside of batch, the numbers of IDs is batch_size/num_instances')

    ############### Model  #######################
    parser.add_argument('--model_arch', default=None, help='Model Architecture')

    ############### Loss  #######################
    parser.add_argument('--softmax_loss', default=None, help='The loss used for classification')
    parser.add_argument('--metric_loss', default=None, help='The loss used as metric loss')
    parser.add_argument("--triplet_margin", default=None, type=float, help='With margin>0 uses normal triplet loss. If margin<=0 or None Soft Margin Triplet Loss is used instead!')
    parser.add_argument('--mean_losses', default=None, help='Use of mixed precision')
    parser.add_argument('--lambda_ce', default=None, type=float, help='multiplier of the classification loss')
    parser.add_argument('--lambda_triplet', default=None, type=float, help='multiplier of the metric loss')

    ############### Optimizer  #######################
    parser.add_argument('--optimizer', default=None, help='Adam or SGD')
    parser.add_argument('--initial_lr', default=None, type=float, help='Initial learning rate after warm-up')


    ############### GPU  #######################
    parser.add_argument('--parallel', default=None, help='Whether to used DataParallel for multi-gpu in one device')    
    parser.add_argument('--half_precision', default=None, help='Use of mixed precision') 

    
    args = parser.parse_args()

    ### Load hyper parameters
    if args.config:
        with open(args.config, "r") as stream:
            data = yaml.safe_load(stream)
    else:
        with open("./config/config.yaml", "r") as stream:
            data = yaml.safe_load(stream)


    ############# DataSet ################
    data['BATCH_SIZE'] = args.batch_size or data['BATCH_SIZE']
    data['p_hflip'] = args.hflip or data['p_hflip']
    data['y_length'] = args.imgsize_y or data['y_length']
    data['x_length'] = args.imgsize_x or data['x_length']
    data['p_rerase'] = args.randomerase or data['p_rerase']
    data['dataset'] = args.dataset or data['dataset']
    data['NUM_INSTANCES'] = args.num_instances or data['NUM_INSTANCES']

    #############  Model  ####################
    data['model_arch'] = args.model_arch or data['model_arch']

    #############  Loss  ####################
    data['triplet_margin'] = args.triplet_margin or data['triplet_margin']
    data['softmax_loss'] = args.softmax_loss or data['softmax_loss']
    data['metric_loss'] = args.metric_loss or data['metric_loss']
    data['ce_loss'] = args.lambda_ce or data['ce_loss']
    data['tri_loss'] = args.lambda_ce or data['tri_loss']
    data['triplet_norm'] = data['triplet_norm']
    data['mining'] = data['mining']
    if args.mean_losses is not None: data['mean_losses'] = bool(args.mean_losses)


    ##############  Optimizer  ##################
    data['optimizer'] = args.optimizer or data['optimizer']
    data['lr'] = args.initial_lr or data['lr']


    #############   Other  ####################
    data['parallel'] = args.parallel or data['parallel']
    data['half_precision'] = args.half_precision or data['half_precision']



    ######### Set Seed for consistent and deterministic results ##############
    set_seed(data['torch_seed'])

    ########### Config print ##############
    print("\n\n\n  Config used: \n")
    print(data)
    print("\n\n\n End config")

    #### Transformation augmentation#  ###############
    teste_transform = transforms.Compose([
                    transforms.Resize((data['y_length'],data['x_length']), antialias=True),
                    transforms.Normalize(data['n_mean'], data['n_std']),

    ])                  
    train_transform = transforms.Compose([
                    transforms.Resize((data['y_length'],data['x_length']), antialias=True),
                    transforms.Pad(10),
                    transforms.RandomCrop((data['y_length'], data['x_length'])),
                    transforms.RandomHorizontalFlip(p=data['p_hflip']),
                    transforms.Normalize(data['n_mean'], data['n_std']),
                    transforms.RandomErasing(p=data['p_rerase'], value=0),
    ])        

    ### GPU ################
    if not data['parallel']:  
        if data['gpu']:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(data['gpu'])

    if data['half_precision']:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = False

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    ################# Dataset ###########################
    if data['dataset']== "VehicleID":
        data_q = CustomDataSet4VehicleID(r'D:\Projects\MBR\data\VehicleID\VehicleID\train_test_split\test_list_1600.txt', data['ROOT_DIR'], is_train=False, mode="q", transform=teste_transform)
        data_g = CustomDataSet4VehicleID(r'D:\Projects\MBR\data\VehicleID\VehicleID\train_test_split\test_list_1600.txt', data['ROOT_DIR'], is_train=False, mode="g", transform=teste_transform)
        data_train = CustomDataSet4VehicleID(r"D:\Projects\MBR\data\VehicleID\VehicleID\train_test_split\train_list.txt", data['ROOT_DIR'], is_train=True, transform=train_transform)
        data_train = DataLoader(data_train, sampler=RandomIdentitySampler(data_train, data['BATCH_SIZE'], data['NUM_INSTANCES']), num_workers=data['num_workers_train'], batch_size = data['BATCH_SIZE'], collate_fn=train_collate_fn, pin_memory=True)#
        data_q = DataLoader(data_q, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])
        data_g = DataLoader(data_g, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])
    if data['dataset']== 'VERIWILD':
        data_q = CustomDataSet4VERIWILD(r'D:\Projects\MBR\data\VeRI-Wild\VERI-WILD\data\train_test_split\test_3000_query.txt', data['ROOT_DIR'], transform=teste_transform, with_view=False)
        data_g = CustomDataSet4VERIWILD(r'D:\Projects\MBR\data\VeRI-Wild\VERI-WILD\data\train_test_split\test_3000.txt', data['ROOT_DIR'], transform=teste_transform, with_view=False)
        data_train = CustomDataSet4VERIWILD(r'D:\Projects\MBR\data\VeRI-Wild\VERI-WILD\data\train_test_split\train_list.txt', data['ROOT_DIR'], transform=train_transform, with_view=False)
        data_train = DataLoader(data_train,
                                sampler=RandomIdentitySampler(data_train, data['BATCH_SIZE'], data['NUM_INSTANCES']),
                                num_workers=data['num_workers_train'], batch_size=data['BATCH_SIZE'],
                                collate_fn=train_collate_fn, pin_memory=True)
        data_q = DataLoader(data_q, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])
        data_g = DataLoader(data_g, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])

    if data['dataset'] == 'Veri776':
        data_q = CustomDataSet4Veri776_withviewpont(data['query_list_file'], data['query_dir'], data['train_keypoint'], data['test_keypoint'], is_train=False, transform=teste_transform)
        data_g = CustomDataSet4Veri776_withviewpont(data['gallery_list_file'], data['teste_dir'], data['train_keypoint'], data['test_keypoint'], is_train=False, transform=teste_transform)
        if data["LAI"]:
            data_train = CustomDataSet4Veri776_withviewpont(data['train_list_file'], data['train_dir'], data['train_keypoint'], data['test_keypoint'], is_train=True, transform=train_transform)
        else:
            data_train = CustomDataSet4Veri776(data['train_list_file'], data['train_dir'], is_train=True, transform=train_transform)
        data_train = DataLoader(data_train, sampler=RandomIdentitySampler(data_train, data['BATCH_SIZE'], data['NUM_INSTANCES']), num_workers=data['num_workers_train'], batch_size = data['BATCH_SIZE'], collate_fn=train_collate_fn, pin_memory=True)
        data_q = DataLoader(data_q, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])
        data_g = DataLoader(data_g, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])


    ##################  Model ############################
    model = get_model(data, device)
    print(model)
    print("model_parameters:" + str(count_parameters(model)))

    if data['parallel']:
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
        print("\n \n Parallel activated!\nDo not use this with LBS!\nIt may result in weird behaviour sometimes.")

    ### Losses ###
    loss_fn = nn.CrossEntropyLoss(label_smoothing=data['label_smoothing'])
    metric_loss = triplet_loss_fastreid(data['triplet_margin'], norm_feat=data['triplet_norm'], mining = data['mining'])

    
    #### Optimizer #####
    optimizer = make_optimizer(data['optimizer'],
                            model,
                            data['lr'],
                            data['weight_decay'],
                            data['bias_lr_factor'],
                            data['momentum'])              #data['eps'])

    ### Schedule for the optimizer
    if data['epoch_freeze_L1toL3'] == 0:                 
        scheduler = make_warmup_scheduler(data['sched_name'],
                                        optimizer,
                                        data['num_epochs'],
                                        data['milestones'],
                                        data['gamma'],
                                        data['warmup_factor'],
                                        data['warmup_iters'],
                                        data['warmup_method'],
                                        last_epoch=-1,
                                        min_lr = data['min_lr']
                                        )
    else:
        scheduler = None


    ### Initiate a Logger with TensorBoard to store Scalars, Embeddings and the weights of the model
    logger = Logger(data)
    logger.save_model(model)

    ##freeze backbone at warmupup epochs up to data['warmup_iters'] 
    if data['freeze_backbone_warmup']:
        for param in model.modelup2L3.parameters():
            param.requires_grad = False
        for param in model.modelL4.parameters():
            param.requires_grad = False
    if data['epoch_freeze_L1toL3'] > 0:
        ### Freeze up to the penultimate layer    
        for param in model.modelup2L3.parameters():
            param.requires_grad = False
        print("\nFroze Backbone before branches!")

    ## Training Loop
    for epoch in tqdm(range(data['num_epochs'])):
        ##unfreeze backbone
        if epoch == data['warmup_iters'] -1: 
            for param in model.modelup2L3.parameters():
                param.requires_grad = True
            for param in model.modelL4.parameters():
                param.requires_grad = True

        if epoch == data['epoch_freeze_L1toL3']-1:
            scheduler = make_warmup_scheduler(data['sched_name'],
                                            optimizer,
                                            data['num_epochs'],
                                            data['milestones'],
                                            data['gamma'],
                                            data['warmup_factor'],
                                            data['warmup_iters'],
                                            data['warmup_method'],
                                            last_epoch=-1,
                                            min_lr = data['min_lr']
                                            )
            for param in model.modelup2L3.parameters():
                param.requires_grad = True
            print("\nUnfrozen Backbone before branches!")
        
        ############  step schedule ##########
        if epoch >= data['epoch_freeze_L1toL3']-1:              
            scheduler.step()

        ################ Train ##################
        train_loss, c_loss, t_loss = train_epoch(model, device, data_train, loss_fn, metric_loss, optimizer, data, logger, epoch, scheduler, scaler)


        ################ Evaluation ##################
        if epoch % data['validation_period'] == 0 or epoch >= data['num_epochs'] - 15:
            if data['re_ranking']:
                cmc, mAP, cmc_re, mAP_re = test_epoch(model, device, data_q, data_g, data['model_arch'],data['re_ranking'], logger, epoch,
                                                      remove_junk=True, scaler=scaler)
                print(
                    '\n EPOCH {}/{} \t train loss {} \t Classification loss {} \t Triplet loss {} \t mAP {} \t CMC1 {} \t CMC5 {}'.format(
                        epoch + 1, data['num_epochs'], train_loss, c_loss, t_loss, mAP, cmc[0], cmc[4]))
                print(
                    '\n EPOCH {}/{} \t train loss {} \t Classification loss {} \t Triplet loss {} \t mAP_re {} \t CMC1_re {} \t CMC5_re {}'.format(
                        epoch + 1, data['num_epochs'], train_loss, c_loss, t_loss, mAP_re, cmc_re[0], cmc_re[4]))
                logger.save_model(model)
            else:
                cmc, mAP = test_epoch(model, device, data_q, data_g, data['model_arch'], data['re_ranking'], logger, epoch,
                                                  remove_junk=True, scaler=scaler)
                print(
                    '\n EPOCH {}/{} \t train loss {} \t Classification loss {} \t Triplet loss {} \t mAP {} \t CMC1 {} \t CMC5 {}'.format(
                        epoch + 1, data['num_epochs'], train_loss, c_loss, t_loss, mAP, cmc[0], cmc[4]))





    #############  Result ################
    print("Best mAP: ", np.max(logger.logscalars['Accuraccy/mAP']))
    print("Best CMC1: ", np.max(logger.logscalars['Accuraccy/CMC1']))
    print("Best CMC5: ", np.max(logger.logscalars['Accuraccy/CMC5']))

    print("Best mAP_re: ", np.max(logger.logscalars['Accuraccy/mAP_re']))
    print("Best CMC1_re: ", np.max(logger.logscalars['Accuraccy/CMC1_re']))
    print("Best CMC5_re: ", np.max(logger.logscalars['Accuraccy/CMC5_re']))

    logger.save_log()   
