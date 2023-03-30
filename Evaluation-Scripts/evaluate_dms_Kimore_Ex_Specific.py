import pandas as pd
import os
import numpy as np
import time

import ast
import pickle

from scipy import signal

def get_lstm_data(datadir, n_epochs=0):
    files_list = sorted(os.listdir(datadir), key=lambda x:int(x.split('.')[0].split('_')[-1]))
    preds = []
    labels = []
    
    if n_epochs!=0:
        n_epochs=(n_epochs//20)*20  
    else:
        epoch_counts = []
        for x in files_list:   
            eph_cnts = [0]   
            with open(datadir+x, 'r') as f:
                for line in f:
                    if line.startswith('y_pred :'):
                        cnt=eph_cnts[-1]+20
                        eph_cnts.append(cnt)
            epoch_counts.append(eph_cnts)  
        n_epochs = min(x[-1] for x in epoch_counts)
              
    for x in files_list:
        pred_list = []
        gtdm_list = []
        count = 0
        with open(datadir+x, 'r') as f:
            for line in f:
                if count==n_epochs:
                    break
                else:
                    if line.startswith('y_pred :'):
                        pred_list.append(line)
                        count+=20
                    if line.startswith('labels :'):
                        gtdm_list.append(line)
                    
        tpreds = ast.literal_eval(pred_list[-1][9:-1])
        tlabels = ast.literal_eval(gtdm_list[-1][9:-1])
        preds.extend(tpreds)
        labels.extend(tlabels)
    
    return preds, labels, n_epochs

def generate_results_countmodels(preds, labels):
    counts_data = list(zip([round(x,0) for x in labels], [round(x,0) for x in preds]))
    mse = sum([(x[0]-x[1])**2 for x in counts_data])/len(counts_data)
    mae = sum([abs(x[0]-x[1]) for x in counts_data])/len(counts_data)
    acc = sum([1 for x in counts_data if abs(x[0]-x[1])==0])/len(counts_data)
    obo = sum([1 for x in counts_data if abs(x[0]-x[1])<=1])/len(counts_data)
    results_dict = {'mse':round(mse, 5), 'acc':round(acc, 5), 'mae':round(mae, 5), 'obo':round(obo, 5)}
    return results_dict

def counts_from_dm_v2(preds, labels, prominence=0.001):
    c_labels = []
    c_preds = []
    c_refs = []
    for x in range(len(preds)):        
        peaks_labels, _ = signal.find_peaks(labels[x], prominence=prominence)
        peaks_preds, _ = signal.find_peaks(preds[x], prominence=prominence)
        c_labels.append(len(peaks_labels))
        c_preds.append(len(peaks_preds))
        c_refs.append(round(sum(labels[x]),0))
    return c_preds, c_labels, c_refs

def smoothen_densitymaps(dms, window_length=29, polyorder=4):
    dms_out = []
    for x in dms:
        dms_out.append(signal.savgol_filter(x, window_length=window_length, polyorder=polyorder).tolist())
    return dms_out

def startends_from_dms_v3(preds, labels, prominence=0.05):
    labels_startends = []
    preds_startends = []
    for x in range(len(preds)): 
        temp_label = [float(i)/max(labels[x]) for i in labels[x]]
        peaks_label, _ = signal.find_peaks(temp_label, prominence=prominence)
        vllys_label, _ = signal.find_peaks([-1*z for z in temp_label], prominence=prominence)
        
        if len(peaks_label)==0 or len(vllys_label)==0:
            labels_startends.append([0, len(labels[x])])
        else:
            l_start_point, l_end_point = 1, len(temp_label)-1
            for y in range(1, peaks_label[0]):
                if abs(temp_label[y+1]-temp_label[y])<1e-5:
                    l_start_point=y
            for y in range(len(temp_label)-1, peaks_label[-1]+1, -1):
                if abs(temp_label[y]-temp_label[y-1])<1e-5:
                    l_end_point=y
            l_startend_list = vllys_label.tolist()
            if abs(l_startend_list[0] - l_start_point)>10:
                l_startend_list = [l_start_point]+l_startend_list
            if abs(l_startend_list[-1] - l_end_point)>10:
                l_startend_list = l_startend_list+[l_end_point]
                
            #print(peaks_label, vllys_label, l_startend_list)
                
            l_start_list = l_startend_list[:-1]
            l_end_list = l_startend_list[1:]
            l_startend_list_full = []
            for z in range(len(l_start_list)):
                l_startend_list_full.append(l_start_list[z])
                l_startend_list_full.append(l_end_list[z])            
            labels_startends.append(l_startend_list_full)

        temp_pred = [float(i)/max(preds[x]) for i in preds[x]]
        peaks_preds, _ = signal.find_peaks(temp_pred, prominence=prominence)     
        vllys_preds, _ = signal.find_peaks([-1*z for z in temp_pred], prominence=prominence)
        
        if len(peaks_preds)==0 or len(vllys_preds)==0:
            preds_startends.append([0, len(preds[x])])
        else: 
            p_start_point, p_end_point = 1, len(temp_pred)-1
            for y in range(1, peaks_preds[0]):
                if abs(temp_pred[y+1]-temp_pred[y])<1e-5:
                    p_start_point=y
            for y in range(len(temp_label)-1, peaks_preds[-1]+1, -1):
                if abs(temp_pred[y]-temp_pred[y-1])<1e-5:
                    p_end_point=y
            p_startend_list = vllys_preds.tolist()
            if abs(p_startend_list[0] - p_start_point)>10:
                p_startend_list = [p_start_point]+p_startend_list
            if abs(p_startend_list[-1] - p_end_point)>10:
                p_startend_list = p_startend_list+[p_end_point]
                
            p_start_list = p_startend_list[:-1]
            p_end_list = p_startend_list[1:]
            p_startend_list_full = []
            for z in range(len(p_start_list)):
                p_startend_list_full.append(p_start_list[z])
                p_startend_list_full.append(p_end_list[z])            
            preds_startends.append(p_startend_list_full)
        
    return preds_startends, labels_startends

def average_frame_difference(preds, refs):
    score_list = []
    for i in range(len(refs)):
        if len(preds[i])==0 or len(refs[i])==0:
            score_list.append(0)
        else:
            if len(preds[i])==len(refs[i]):
                temp_list = [abs(preds[i][x]-refs[i][x]) for x in range(len(preds[i]))]
                score = sum(temp_list)/len(preds[i])
            elif len(preds[i])>len(refs[i]):
                all_scores = []
                for z in range(len(preds[i])-len(refs[i])):
                    temp_list = [abs(preds[i][x+z]-refs[i][x]) for x in range(len(refs[i]))]
                    score = sum(temp_list)/len(refs[i])
                    all_scores.append(score)
                score = min(all_scores)
            else:
                all_scores = []
                for z in range(len(refs[i])-len(preds[i])+1):
                    temp_list = [abs(preds[i][x]-refs[i][x+z]) for x in range(len(preds[i]))]
                    score = sum(temp_list)/len(preds[i])
                    all_scores.append(score)
                score = min(all_scores)
            score_list.append(score)
    return score_list

def iou_frames(preds, refs):
    scores_list = []
        
    for i in range(len(preds)):
        if len(preds[i])==0 or len(refs[i])==0:
            scores_list.append(0)    
        else:  
            p_start_list = preds[i][::2]
            p_ends_list = preds[i][1::2]
            l_start_list = refs[i][::2]
            l_ends_list = refs[i][1::2]
            
            #print(len(p_start_list), len(p_ends_list))

            if len(p_start_list)==len(l_start_list):
                area_union = 0
                area_insec = 0
                for x in range(len(l_start_list)):
                    area_union_rep = 0
                    area_insec_rep = 0
                    for y in range(1, 2000):
                        if (y>=min(p_start_list[x], l_start_list[x]) and y<=max(p_ends_list[x], l_ends_list[x])):
                            area_union_rep+=1
                        if (y>=max(p_start_list[x], l_start_list[x]) and y<=min(p_ends_list[x], l_ends_list[x])):
                            area_insec_rep+=1
                    area_union+=area_union_rep
                    area_insec+=area_insec_rep
                if area_union!=0:
                    score_iou = area_insec/area_union
                else:
                    score_iou = 0

            elif len(p_start_list)>len(l_start_list):
                all_scores = []
                for z in range(len(p_start_list)-len(l_start_list)):
                    area_union = 0
                    area_insec = 0
                    for x in range(len(l_start_list)):
                        area_union_rep = 0
                        area_insec_rep = 0
                        for y in range(1, 2000):
                            if (y>=min(p_start_list[x+z], l_start_list[x]) and y<=max(p_ends_list[x+z], l_ends_list[x])):
                                area_union_rep+=1
                            if (y>=max(p_start_list[x+z], l_start_list[x]) and y<=min(p_ends_list[x+z], l_ends_list[x])):
                                area_insec_rep+=1
                        area_union+=area_union_rep
                        area_insec+=area_insec_rep
                    if area_union!=0:
                        z_score_iou = area_insec/area_union
                    else:
                        z_score_iou = 0
                    all_scores.append(z_score_iou)
                score_iou = max(all_scores)

            else:
                all_scores = []
                for z in range(len(l_start_list)-len(p_start_list)):
                    #print(len(preds[i]), len(refs[i]))
                    area_union = 0
                    area_insec = 0
                    for x in range(len(p_start_list)):
                        area_union_rep = 0
                        area_insec_rep = 0
                        #print(x, x+z)
                        for y in range(1, 2000):
                            if (y>=min(p_start_list[x], l_start_list[x+z]) and y<=max(p_ends_list[x], l_ends_list[x+z])):
                                area_union_rep+=1
                            if (y>=max(p_start_list[x], l_start_list[x+z]) and y<=min(p_ends_list[x], l_ends_list[x+z])):
                                area_insec_rep+=1
                        area_union+=area_union_rep
                        area_insec+=area_insec_rep
                    if area_union!=0:
                        z_score_iou = area_insec/area_union
                    else:
                        z_score_iou = 0
                    all_scores.append(z_score_iou)
                score_iou = max(all_scores)
            scores_list.append(score_iou)
        
    return scores_list

def generate_results_dm_models(preds, labels, debug=False):
    preds_sm = smoothen_densitymaps(preds)    
    labels_sm = smoothen_densitymaps(labels)
    preds_ct, labels_ct, refs_ct = counts_from_dm_v2(preds_sm, labels_sm)
    counts_results_dict = generate_results_countmodels(preds_ct, refs_ct)
    
    preds_sm_dm = smoothen_densitymaps(preds, window_length=29, polyorder=2)    
    labels_sm_dm = smoothen_densitymaps(labels, window_length=29, polyorder=2)
    
    preds_se, labels_se = startends_from_dms_v3(preds_sm_dm, labels_sm_dm)
    afd = np.mean((average_frame_difference(preds_se, labels_se)))
    iou = np.mean((iou_frames(preds_se, labels_se)))
    dm_results_dict = {'afd':round(afd, 5), 'iou':round(iou, 5)}
    results_dict = {**counts_results_dict, **dm_results_dict}
    if debug==False:
        return results_dict, []
    else:
        counts_results_dict = generate_results_countmodels(refs_ct, labels_ct)
        return results_dict, counts_results_dict
    
def load_dms_from_video_dir(data_dir):
    dm_labels = []
    for x in range(1, len(os.listdir(data_dir))+1):
        with open(data_dir+'V'+str(x)+'_DM.csv', "r") as f: 
            data_dm = f.read()
            data_dm = [float(x) for x in data_dm.split("\n")[0].split(",")]
            dm_labels.append((x, data_dm))
    dms = [x[1] for x in dm_labels]
    tags = []
    for x in [x[0] for x in dm_labels]:
        if int(x)<=72:
            tags.append('Ex0_V'+str(x))
        elif int(x)>72 and int(x)<=72+65:
            tags.append('Ex1_V'+str(x))
        elif int(x)>72+65 and int(x)<=72+65+68:
            tags.append('Ex2_V'+str(x))
        elif int(x)>72+65+68 and int(x)<=72+65+68+72:
            tags.append('Ex3_V'+str(x))
        elif int(x)>72+65+68+72:
            tags.append('Ex4_V'+str(x))    
    return dms, tags

def load_dms_from_mappedfile(file_path):
    with open(file_path, 'rb') as f:
        ref_data = pickle.load(f)
    return [x[1] for x in ref_data], [x[0] for x in ref_data]

def load_dms_from_masterdataset(file_path):
    md_labels = []
    with open(file_path, "r") as f:
        md_data = f.readlines()
    for x in md_data:
        video_idx = x.split(':')[0]
        data_dm = [float(z) for z in x.split(':')[-2].lstrip('[').rstrip(']').split(",")]
        md_labels.append((video_idx, data_dm))
        
    dms = [x[1] for x in md_labels]
    tags = []
    for x in [x[0].strip('V') for x in md_labels]:
        if int(x)<=72:
            tags.append('Ex0_V'+str(x))
        elif int(x)>72 and int(x)<=72+65:
            tags.append('Ex1_V'+str(x))
        elif int(x)>72+65 and int(x)<=72+65+68:
            tags.append('Ex2_V'+str(x))
        elif int(x)>72+65+68 and int(x)<=72+65+68+72:
            tags.append('Ex3_V'+str(x))
        elif int(x)>72+65+68+72:
            tags.append('Ex4_V'+str(x)) 
               
    return dms, tags

def match_dms_using_correlate(preds_logs, labels_logs, labels_true, labels_tags):
    labels_logs_x = []
    for x in labels_logs:
        normalized_factor = np.sqrt(sum(np.square(np.array(x))))
        new_sig = x/normalized_factor
        labels_logs_x.append(new_sig)
    labels_true_x = []
    for x in labels_true:
        normalized_factor = np.sqrt(sum(np.square(np.array(x))))
        new_sig = x/normalized_factor
        labels_true_x.append(new_sig)
        
    matched_data = []
    for x in range(len(labels_logs)):
        max_corr = 0
        max_corr_idx = 0
        for z in range(len(labels_true)):
            corr = signal.correlate(labels_logs_x[x], labels_true_x[z], mode='full')
            if max(corr)>max_corr:
                max_corr=max(corr)
                max_corr_idx = z
        #print(max_corr, max_corr_idx)
        matched_data.append((preds_logs[x], labels_logs[x],
                             labels_true[max_corr_idx],
                             labels_tags[max_corr_idx]))
    #print([y[-1] for y in matched_data])
    return matched_data

def match_dms_using_idx(preds_logs, labels_logs, labels_true, labels_tags):
    return list(zip(preds_logs, labels_logs, labels_true, labels_tags))
        
def generate_results_exercises(m_data):
    results_all = []
    preds_x = {0:[], 1:[], 2:[], 3:[], 4:[]}
    labels_x = {0:[], 1:[], 2:[], 3:[], 4:[]}
    for x in m_data:
        if x[-1].split('_')[0]=='Ex0':
            preds_x[0].append(x[0])
            labels_x[0].append(x[1])
        elif x[-1].split('_')[0]=='Ex1':
            preds_x[1].append(x[0])
            labels_x[1].append(x[1])
        elif x[-1].split('_')[0]=='Ex2':
            preds_x[2].append(x[0])
            labels_x[2].append(x[1])
        elif x[-1].split('_')[0]=='Ex3':
            preds_x[3].append(x[0])
            labels_x[3].append(x[1])
        elif x[-1].split('_')[0]=='Ex4':
            preds_x[4].append(x[0])
            labels_x[4].append(x[1])
    results_all.append(('Ex_1', generate_results_dm_models(preds_x[0], labels_x[0])))
    results_all.append(('Ex_2', generate_results_dm_models(preds_x[1], labels_x[1])))
    results_all.append(('Ex_3', generate_results_dm_models(preds_x[2], labels_x[2])))
    results_all.append(('Ex_4', generate_results_dm_models(preds_x[3], labels_x[3])))
    results_all.append(('Ex_5', generate_results_dm_models(preds_x[4], labels_x[4])))
    
    return results_all 

def main():
    data_dir_logs = 'logs/'
    data_path_mapped_file = 'Others/dm_mapped_v1.pkl'
    data_path_md_file = 'MD/master_dataset.txt'
    data_dir_video_dms = 'DMs/'
    
    print("Choose 0 for Using Mapped File, 1 for MasterDataset, 2 for Video-dir DMs: ")
    videodir_or_mapfile = int(str(input()))
    
    if videodir_or_mapfile==0:
        dms_true, tags_true = load_dms_from_mappedfile(data_path_mapped_file)
    elif videodir_or_mapfile==1:
        dms_true, tags_true = load_dms_from_masterdataset(data_path_md_file)
    elif videodir_or_mapfile==2:
        dms_true, tags_true = load_dms_from_video_dir(data_dir_video_dms)
    
    print("Please enter number of epochs (0 for last epoch with results): ")
    n_epochs = int(str(input()))
    
    preds_dm, labels_dm, n_epochs = get_lstm_data(data_dir_logs, n_epochs)
    #print(len(preds_dm), len(labels_dm))
    
    matched_data = match_dms_using_idx(preds_dm, labels_dm, dms_true, tags_true)
    
    results_dm, results_debug = generate_results_dm_models(preds_dm, labels_dm, True)
    print("For {} Epochs, the full results are:".format(n_epochs))
    print(results_dm)
    
    results_all = generate_results_exercises(matched_data)
    print("For {} Epochs, the exercise specific results are:".format(n_epochs))
    for res in results_all:
        print(res)
    
    
if __name__=="__main__":
    main()