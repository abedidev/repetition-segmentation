"""train or valid looping """
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from TransRAC_v3 import TransferModel
from KinectFeatureLoader_v3 import KinectBodyJointsDataset

import warnings

# from masterLoader import *

torch.manual_seed(1)  # random seed. We not yet optimization it.


def train_loop(n_folds, n_epochs, model, data_set, batch_size=1, lr=1e-6, device_ids=[0]):
    
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")
    currEpoch = 0
    #trainloader = DataLoader(train_set, batch_size=batch_size, pin_memory=False, shuffle=True, num_workers=16)
    
    total_size = len(data_set)
    fraction = 1/n_folds
    seg = int(total_size * fraction)
    
    for x_fold in range(n_folds):
        trll = 0
        trlr = x_fold * seg
        vall = trlr
        valr = (x_fold * seg) + seg
        trrl = valr
        trrr = total_size
        # print("train indices: [%d,%d),[%d,%d), test indices: [%d,%d)" 
        #       % (trll,trlr,trrl,trrr,vall,valr))
        
        train_lf_idx = list(range(trll,trlr))
        train_rt_idx = list(range(trrl,trrr))
        
        train_idx = train_lf_idx + train_rt_idx
        valid_idx = list(range(vall,valr))
        
        train_set = torch.utils.data.dataset.Subset(data_set, train_idx)
        valid_set = torch.utils.data.dataset.Subset(data_set, valid_idx)
        
        # print(len(train_set),len(val_set))
        # print()
        
        trainloader = DataLoader(train_set, batch_size=batch_size, pin_memory=False, shuffle=True, num_workers=0)
        validloader = DataLoader(valid_set, batch_size=batch_size, pin_memory=False, shuffle=True, num_workers=0)
        
        model = nn.DataParallel(model.to(device), device_ids=device_ids)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        milestones = [i for i in range(0, n_epochs, 20)]
        scheduler_a = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, cooldown=3)
        scheduler_b = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.8)  # three step decay

        # writer = SummaryWriter(log_dir=os.path.join('log/', log_dir))
        scaler = GradScaler()

        # if lastckpt is not None:
        # print("loading checkpoint")
        # checkpoint = torch.load(lastckpt)
        # currEpoch = checkpoint['epoch']
        # # # # load hyperparameters by pytorch
        # # # # if change model
        # # net_dict=model.state_dict()
        # # state_dict={k: v for k, v in checkpoint.items() if k in net_dict.keys()}
        # # net_dict.update(state_dict)
        # # model.load_state_dict(net_dict, strict=False)

        # # # # or don't change model
        # model.load_state_dict(checkpoint['state_dict'], strict=False)

        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # del checkpoint

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        lossMSE = nn.MSELoss()
        lossSL1 = nn.SmoothL1Loss()
        
        lossKLD = nn.KLDivLoss(reduction = 'batchmean', log_target=True).to(device)

        t_losses_df = pd.DataFrame(columns=['Epoch', 'Train_loss', 'Train_MAE', 'Train_OBO'])
        v_losses_df = pd.DataFrame(columns=['Epoch', 'Valid_loss', 'Valid_MAE', 'Valid_OBO'])    
        
        train_file_names = []
        valid_file_names = []
        
        for epoch in tqdm(range(currEpoch, n_epochs + currEpoch)):
            trainLosses = []
            validLosses = []
            trainLoss1 = []
            validLoss1 = []
            trainOBO = []
            validOBO = []
            trainMAE = []
            validMAE = []
            predCount = []
            Count = []
            results = {'video_path':[],'pred_dm': [], 'actual_dm': []}

            pbar = tqdm(trainloader, total=len(trainloader))
            batch_idx = 0
            for video_path, input, target in pbar:
                train_file_names.extend(list(video_path))
                with autocast():
                    model.train()
                    optimizer.zero_grad()
                    acc = 0
                    input = input.type(torch.FloatTensor).to(device)
                    density = target.type(torch.FloatTensor).to(device)
                    count = torch.sum(target, dim=1).type(torch.FloatTensor).round().to(device)
                    output, matrixs = model(input)
                    predict_count = torch.sum(output, dim=1).type(torch.FloatTensor).to(device)
                    predict_density = output
                    t_loss1 = lossMSE(predict_density, density)
                    t_loss2 = lossSL1(predict_count, count)
                    # loss2 = lossMSE(predict_count, count)
                    t_loss3 = torch.sum(torch.div(torch.abs(predict_count - count), count + 1e-1)) / \
                        predict_count.flatten().shape[0]  # mae
                        
                    counts = count.unsqueeze(-1)
                    pred_counts = predict_count.unsqueeze(-1)
                    #ce_target = torch.div(density, counts)
                    #ce_target = ce_target.float()
                    #output_ls = nn.functional.log_softmax(output, dim=-1)
                    output_ls = torch.log(output)
                    kld_target = torch.div(density, counts)
                    target_ls = nn.functional.log_softmax(kld_target, dim=-1)
                    #target_ls = torch.log(kld_target)
                    t_loss4 = 0.5*lossKLD(output_ls, target_ls)
                        
                    #print(output_ls.cpu().detach().numpy())
                    #print(target_ls.cpu().detach().numpy())
                    #print(counts.cpu().detach().numpy())
                    #print(pred_counts.cpu().detach().numpy())
                        
                        
                    #print((loss1.cpu().detach().item(), loss2.cpu().detach().item(), 
                    #       loss3.cpu().detach().item(), loss4.cpu().detach().item()))
                        
                    #print(loss4)
                    t_loss = t_loss1
                    #loss = loss4 + 100*loss1 + 10*loss1*loss2
                    #loss = (1000*loss1)*(1 + loss2/500 + loss3/50)
                    #loss = (1000*loss1)*(1+ loss4/200)
                    #loss = loss4
                    #loss = 100*(loss1+loss4)*(1+loss2/10)
                            
                    results['video_path'].append(video_path)
                    results['pred_dm'].append(output.cpu().detach().numpy())
                    results['actual_dm'].append(target.cpu().detach().numpy())
                                                            

                    # calculate MAE or OBO
                    gaps = torch.sub(predict_count, count).reshape(-1).cpu().detach().numpy().reshape(-1).tolist()
                    for item in gaps:
                        if abs(item) <= 1:
                            acc += 1
                    #OBO = acc / predict_count.flatten().shape[0]
                    T_OBO = acc / count.flatten().shape[0]
                    #print(count)
                    trainOBO.append(T_OBO)
                    T_MAE = t_loss3.item()
                    trainMAE.append(T_MAE)

                    train_loss = t_loss.item()
                    train_loss1 = t_loss1.item()
                    trainLosses.append(train_loss)
                    trainLoss1.append(train_loss1)
                    batch_idx += 1
                    t_loss_data = {'Epoch': epoch, 'loss_train': train_loss, 'loss_mse': t_loss1.item(),
                                   'loss_sl1': t_loss2.item(), 'loss_kld': t_loss4.item(), 
                                   'Train MAE': T_MAE, 'Train OBO ': T_OBO}
                    pbar.set_postfix(t_loss_data)

                        # if batch_idx % 10 == 0:
                        # writer.add_scalars('train/loss',
                        #    {"loss": np.mean(trainLosses)},
                        #    epoch * len(trainloader) + batch_idx)
                        # writer.add_scalars('train/MAE',
                        #    {"MAE": np.mean(trainMAE)},
                        #    epoch * len(trainloader) + batch_idx)
                        # writer.add_scalars('train/OBO',
                        #    {"OBO": np.mean(trainOBO)},
                        #    epoch * len(trainloader) + batch_idx)

                scaler.scale(t_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # add loss after certain num of epochs to losses dataframe
            t_losses_df = t_losses_df.append({'Epoch': epoch, 'Train_loss': np.mean(trainLoss1),
                                            'Train_MAE': np.mean(trainMAE),'Train_OBO': np.mean(T_OBO)},
                                           ignore_index=True)
            
            predicted_rep_counts = [round(sum(results['pred_dm'][i][0])) for i in range(len(results['pred_dm']))]
            ground_truth_rep_counts = [round(sum(results['actual_dm'][i][0])) for i in range(len(results['actual_dm']))]
            #print(predicted_rep_counts[::5], ground_truth_rep_counts[::5])
            

            with torch.no_grad():
                batch_idx = 0
                pbar = tqdm(validloader, total=len(validloader))
                for video_path, input, target in pbar:
                    valid_file_names.extend(list(video_path))
                    model.eval()
                    acc = 0
                    input = input.type(torch.FloatTensor).to(device)
                    density = target.type(torch.FloatTensor).to(device)
                    count = torch.sum(target, dim=1).type(torch.FloatTensor).round().to(device)

                    output, sim_matrix = model(input)
                    predict_count = torch.sum(output, dim=1).type(torch.FloatTensor).to(device)
                    predict_density = output

                    v_loss1 = lossMSE(predict_density, density)
                    v_loss2 = lossSL1(predict_count, count)
                    # loss2 = lossMSE(predict_count, count)
                    v_loss3 = torch.sum(torch.div(torch.abs(predict_count - count), count + 1e-1)) / \
                                predict_count.flatten().shape[0]  # mae
                                
                    v_counts = count.unsqueeze(-1)
                    pred_counts = predict_count.unsqueeze(-1)
                    #ce_target = torch.div(density, counts)
                    #ce_target = ce_target.float()
                    #output_ls = nn.functional.log_softmax(output, dim=-1)
                    v_output_ls = torch.log(output)
                    v_kld_target = torch.div(density, v_counts)
                    v_target_ls = nn.functional.log_softmax(v_kld_target, dim=-1)
                    #target_ls = torch.log(kld_target)
                    v_loss4 = 0.5*lossKLD(v_output_ls, v_target_ls)            
                                
                    v_loss = v_loss1
                        
                    gaps = torch.sub(predict_count, count).reshape(-1).cpu().detach().numpy().reshape(-1).tolist()
                    for item in gaps:
                        if abs(item) <= 1:
                            acc += 1
                    V_OBO = acc / predict_count.flatten().shape[0]
                    validOBO.append(V_OBO)
                    V_MAE = v_loss3.item()
                    validMAE.append(V_MAE)
                    valid_loss = v_loss.item()
                    valid_loss1 = v_loss1.item()
                    validLosses.append(valid_loss)
                    validLoss1.append(valid_loss1)
                    

                    batch_idx += 1
                    
                    v_loss_data = {'Epoch': epoch, 'loss_valid': valid_loss, 'v_loss_mse': v_loss1.item(),
                                   'v_loss_sl1': v_loss2.item(), 'v_loss_kld': v_loss4.item(), 
                                   'Valid MAE': T_MAE, 'Valid OBO ': T_OBO}                  
                    pbar.set_postfix(v_loss_data)

                    # writer.add_scalars('valid/loss', {"loss": np.mean(validLosses)},
                    #                    epoch)
                    # writer.add_scalars('valid/OBO', {"OBO": np.mean(validOBO)},
                    #                    epoch)
                    # writer.add_scalars('valid/MAE',
                    #    {"MAE": np.mean(validMAE)},
                    #    epoch)
                    
            v_losses_df = v_losses_df.append({'Epoch': epoch, 'Valid_loss': np.mean(trainLoss1),
                                              'Valid_MAE': np.mean(validMAE),'Valid_OBO': np.mean(V_OBO)},
                                             ignore_index=True)
                    
            scheduler_a.step(v_loss)
            scheduler_b.step()
            
            if not os.path.exists('checkpoint/{0}/'.format('ckpt')):
                os.mkdir('checkpoint/{0}/'.format('ckpt'))
            if (epoch<3*(n_epochs//2) and epoch%10==0) or (epoch>(3*n_epochs//2) and epoch%5==0):
                    checkpoint = {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'trainLosses': trainLosses,
                        'valLosses': validLosses
                    }
                    torch.save(checkpoint, 
                               'checkpoint/{0}/'.format('ckpt')+'f'+str(x_fold)+'_'+str(epoch)+'_'+str(round(np.mean(trainLoss1), 5))+'.pt')

            # writer.add_scalars('learning rate', {"learning rate": optimizer.state_dict()['param_groups'][0]['lr']}, epoch)
            # writer.add_scalars('epoch_trainMAE', {"epoch_trainMAE": np.mean(trainMAE)}, epoch)
            # writer.add_scalars('epoch_trainOBO', {"epoch_trainOBO": np.mean(trainOBO)}, epoch)
            # writer.add_scalars('epoch_trainloss', {"epoch_trainloss": np.mean(trainLosses)}, epoch)
            
        train_file_names = list(set(train_file_names))
        valid_file_names = list(set(valid_file_names))
        if not os.path.exists('outputs'):
            os.mkdir('outputs')
        with open('outputs/train_file_names_f{}.txt'.format(str(x_fold)), 'w') as fp:
            for row in train_file_names:
                s = "".join(map(str, row))
                fp.write(s+'\n')
        with open('outputs/valid_file_names_f{}.txt'.format(str(x_fold)), 'w') as fp:
            for row in valid_file_names:
                s = "".join(map(str, row))
                fp.write(s+'\n')
            
        print("Training Finished, CV-Fold-{}".format(str(x_fold)))
        t_losses_df.to_csv('All_train_losses_F{}.csv'.format(str(x_fold)), index=False)
        v_losses_df.to_csv('All_valid_losses_F{}.csv'.format(str(x_fold)), index=False)
    
def main():    
    NUM_FOLDS = 2
    NUM_EPOCHS = 2
    BATCH_SIZE = 8
    LR = 8e-4
    lastckpt = None
    
    
    NUM_FRAME = 512
    PADDING_SIZE = 1280
    SCALES = [1, 4, 8]
    
    my_model = TransferModel(None, SCALES, PADDING_SIZE)
    
    dataset = KinectBodyJointsDataset(root_dir='',exercise_dir='',labels_dir='Annotations',
                                      target_frames=NUM_FRAME, padding_size=PADDING_SIZE)

    train_loop(NUM_FOLDS, NUM_EPOCHS, my_model, dataset, batch_size=BATCH_SIZE, lr=LR)
    
if __name__=="__main__":
    main()