""" test of TransRAC """
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#from masterLoader import *
from tqdm import tqdm
from TransRAC_v3 import *
# from tools.my_tools import paint_smi_matrixs, plot_inference, density_map
import pickle
#from visualisations import plot
from plot_density_map import plotDensityMap
from OpenposeFeatureLoader_v3 import OpenposeBodyJointsDataset

torch.manual_seed(1)


def test_loop(model, test_dataset, inference=True, batch_size=1, lastckpt=None,device_ids=[0], visualise=False, save_name=None):
    device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")
    currEpoch = 0
    # call our dataloader here
    testloader = DataLoader(test_dataset, batch_size=batch_size)
    # testloader = DataLoader(test_set, batch_size=batch_size, pin_memory=False, shuffle=True, num_workers=10)
    model = nn.DataParallel(model.to(device), device_ids=device_ids)
    model.eval()

    if lastckpt is not None:
        checkpoint = torch.load(lastckpt)
        currEpoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        del checkpoint

    results = {'video_path':[],'pred_dm': [], 'actual_dm': []}
    testOBO = []
    testMAE = []
    predCount = []
    Count = []
    if inference:
        with torch.no_grad():
            
            pbar = tqdm(testloader, total=len(testloader))
            for video_path,input,target in pbar:
                model.eval()
                #print(video_path)
                acc = 0
                input = input.to(device)
                count = torch.sum(target, dim=1).round().to(device)
                output, sim_matrix = model(input)
                # kld_target = torch.div(output, count)
                # output_ls = nn.functional.log_softmax(output, dim=-1)
                #try:
                #    with open('test_results/output_{0}.pkl'.format(batch_idx), 'wb') as f:
                #        pickle.dump(output, f)
                #    with open('test_results/simmat_{0}.pkl'.format(batch_idx), 'wb') as f:
                #        pickle.dump(simmat, f)
                #except:
                #    print('failed to save density map and/or sim-matrix')
                 #predict_count = torch.sum(output, dim=1).round()
                  #print('pred: {0} actual: {1}'.format(predict_count.item(), count.item()))
                results['video_path'].append(video_path)
                results['pred_dm'].append(output.cpu().detach().numpy())
                results['actual_dm'].append(target.cpu().detach().numpy())
                # print('Output value: ')
                # print(output)

                if visualise:
                     for i in range(testloader.batch_size):
                          y = output[i].cpu().detach().numpy()
                          print(y, type(y))
                          plotDensityMap(y, 'test_img'+str(i))


                    # mae = torch.sum(torch.div(torch.abs(predict_count - count), count + 1e-1)) / \
                    #      predict_count.flatten().shape[0]  # mae

                    # gaps = torch.sub(predict_count, count).reshape(-1).cpu().detach().numpy().reshape(-1).tolist()
                    # for item in gaps:
                    #     if abs(item) <= 1:
                    #         acc += 1
                    # OBO = acc / predict_count.flatten().shape[0]
                    # testOBO.append(OBO)
                    # MAE = mae.item()
                    # testMAE.append(MAE)

                    # predCount.append(predict_count.item())
                    # Count.append(count.item())
                    # print('predict count :{0}, groundtruth :{1}'.format(predict_count.item(), count.item()))
                
                    # break


        # print("MAE:{0},OBO:{1}".format(np.mean(testMAE), np.mean(testOBO)))
        # plot_inference(predict_count, count)
    if save_name:
        with open(f'{save_name}.pkl', 'wb') as f:
            pickle.dump(results, f)

def main():
    
    NUM_FOLDS = 5
    NUM_EPOCHS = 80
    NUM_FRAME = 512
    PADDING_SIZE = 1280
    SCALES = [1, 4, 8]
    
    my_model = TransferModel(None, SCALES, PADDING_SIZE)
    
    dataset = OpenposeBodyJointsDataset(root_dir='',exercise_dir='openpose_data/',labels_dir='Annotations',
                                      target_frames=NUM_FRAME, padding_size=PADDING_SIZE)
    
    checkpoints = {}
    for x in os.listdir('checkpoint/ckpt/'):
        if x.startswith('f') and int(x.split('_')[1])==NUM_EPOCHS:
            checkpoints[int(x.split('_')[0].strip('f'))] = 'checkpoint/ckpt/'+x
    print(checkpoints)
    
    for x in tqdm(range(NUM_FOLDS)):
        test_loop(model=my_model,test_dataset= dataset,inference= True,batch_size=1,
                lastckpt=checkpoints[x], visualise=False, save_name='agnostic_raw_results_f{}'.format(str(x)))

if __name__=="__main__":
    main()
