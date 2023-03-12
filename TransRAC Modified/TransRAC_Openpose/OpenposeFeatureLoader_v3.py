import math
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pickle

from label_norm import normalize_label


class OpenposeBodyJointsDataset(Dataset):
    def __init__(self, root_dir, exercise_dir, labels_dir, target_frames=128, padding_size=None):
        self.rng = np.random.default_rng(27)
        self.exercise_dir = exercise_dir
        self.labels_dir = labels_dir
        self.target_frames = target_frames
        self.num_frames = None
        self.padding_size = padding_size
        
        self.openpose_data = {}
        for x in os.listdir(self.exercise_dir):
            with open(exercise_dir+x, 'rb') as f:
                self.openpose_data.update(pickle.load(f))
        self.openpose_data_keys = list(self.openpose_data.keys())
        self.rng.shuffle(self.openpose_data_keys)
        
    def __len__(self):
        return len(self.openpose_data_keys)
    
    def __getitem__(self, index):
        features = self.getRawBodyJointFeatures(self.openpose_data[self.openpose_data_keys[index]])
        labels_path = self.getLabelsPath(self.openpose_data_keys[index])
        
        timestamps = self.get_timestamps(labels_path)
        labels = self.preprocess(
            self.num_frames, timestamps, self.num_frames)
               
        features, labels = self.pad_data(features, labels)
        
        features = torch.Tensor(features)
        labels = torch.Tensor(labels)
        
        return labels_path, features, labels
    
    def getLabelsPath(self, data_key):
        
        exerciseNum = int(data_key.split('_')[0][-1])
        videoName = data_key.split('_')[-1]
       
        video_path = os.path.join(self.labels_dir, 'E' + str(exerciseNum), 'csv_files', videoName + '.csv')
        return video_path
    
    def getRawBodyJointFeatures(self, bodyjoints):
        self.num_frames = len(bodyjoints)
        # print(len(lmlist))

        # construct the feature list as [[x1,y1,z1,x2,y2,z2....][...][...]] where xi,yi,zi are the x,y,z coordinates of a body joint/sensor
        raw_features = []
        for frame in bodyjoints:
            frame_features = []
            for landmark in frame:
                x,y,z = landmark
                frame_features += [x,y,z]
            raw_features.append(frame_features)
        # print(raw_features)
        return raw_features 
    
    def pad_data(self, frames, labels):      
        #print(len(frames), len(labels))
              
        if self.padding_size!=None:
            frame_len = len(frames)
            label_len = len(labels)      
            if frame_len>label_len:
                frames[frame_len-label_len:]                
            elif frame_len<label_len:
                frames = np.array(frames)
                frames = np.concatenate([frames, np.tile(frames[-1], (label_len-frame_len, 1))]).tolist()       
            data_len = min([len(frames), len(labels)])
            
            if data_len<self.padding_size:
                frames = np.array(frames)
                frames = np.concatenate([frames, np.tile(np.zeros(frames[-1].shape), (self.padding_size-data_len, 1))]).tolist()
                labels = labels + [0]*(self.padding_size-data_len)    
            if data_len>self.padding_size:
                frames = frames[:self.padding_size]
                labels = labels[:self.padding_size]
                               
        #print(len(frames), len(labels))    
        return frames, labels
    
    def get_timestamps(self, path):
        # read V{num}.csv to RAM
        self.check_file_exist(path)
        FRAME_RATE = 25
        test_labels_df = pd.read_csv(path)

        # print(test_labels_df)

        test_labels_df['start frame num'] = test_labels_df['Start'].apply(
            lambda x: ((int(x.split(":")[0])*3600) + (int(x.split(":")[1])*60) + float(x.split(':')[2])) * FRAME_RATE)

        test_labels_df['end frame num'] = test_labels_df['End'].apply(
            lambda x: ((int(x.split(":")[0])*3600) + (int(x.split(":")[1])*60) + float(x.split(':')[2])) * FRAME_RATE)

        # print('test labels with frame num\n', test_labels_df)

        timestamps = []
        for index, annotation in test_labels_df.iterrows():
            # print(annotation)
            timestamps.append(annotation['start frame num'])
            timestamps.append(annotation['end frame num'])

        # print(timestamps)

        return timestamps
    
    def preprocess(self, video_frame_length, time_points, num_frames):
        """
        process label(.csv) to density map label
        Args:
            video_frame_length: video total frame number, i.e 1024frames
            time_points: label point example [1, 23, 23, 40,45,70,.....] or [0]
            num_frames: 64
        Returns: for example [0.1,0.8,0.1, .....]
        """
        new_crop = []
        for i in range(len(time_points)):  # frame_length -> 64
            item = min(math.ceil(
                (float((time_points[i])) / float(video_frame_length)) * num_frames), num_frames - 1)
            new_crop.append(item)
        new_crop = np.sort(new_crop)
        label = normalize_label(new_crop, num_frames)

        return label
    
    def check_file_exist(self, filename, msg_tmpl='file "{}" does not exist'):
        if not os.path.isfile(filename):
            raise FileNotFoundError(msg_tmpl.format(filename))
            
    def linearSampling(self, frames):
        n_frames = len(frames)
        if self.padding_size!=None:
            if n_frames>self.padding_size:
                return frames[n_frames-self.padding_size:]
            elif self.padding_size>n_frames:
                frames =  np.array(frames)
                return np.concatenate([frames, np.tile(frames[-1], (self.padding_size-n_frames, 1))]).tolist()
            else:
                return frames           
        else:
            return [frames[i * n_frames // self.target_frames]
                    for i in range(self.target_frames)]


# TO-DO:
# 1. Complete Reading CSV files of keyPoints, store keyPoints in a list, find angles for the samples
# frames and distance ratios of joints
# 2. Read these features as a dataLoader item
# 3. Finalise the list of distance ratios and angles for each exercise and code it up in a config file
# 4. Train all the 70-80 videos of all 5 exercises together
# 5. Create a mapping of test inputs and labels from given data
# 6. Output density map and repetition count plot for output with diff loss functions like KL divergence,
# Binary cross entropy


def makePersonIdVideoIdMapping():
    video_person_exercise_mapping = pd.DataFrame(columns=['PersonID', 'ExerciseNum', 'AnnotationVideoNum'])
    # print(video_person_exercise_mapping)
    exercise_dataset_dir = '../Exercises/Exercises'
    for exerciseNum in os.listdir(exercise_dataset_dir):
        exercise_dir = os.path.join(exercise_dataset_dir, exerciseNum, 'videos')
        if os.path.isdir(exercise_dir):
            for videoNum in os.listdir(exercise_dir):
                video_dir = os.path.join(exercise_dir, videoNum)
                if os.path.isdir(video_dir):
                    for fileName in os.listdir(video_dir):
                        if str(fileName).endswith(".xlsx"):
                            personId = fileName.removesuffix(".xlsx")
                            newEntry = {'PersonID': personId, 'ExerciseNum': exerciseNum,
                                        'AnnotationVideoNum': videoNum}
                            # print(newEntry)
                            video_person_exercise_mapping = video_person_exercise_mapping.append(newEntry,
                                                                                                 ignore_index=True)

    # print(video_person_exercise_mapping.head(10))
    video_person_exercise_mapping.to_csv('video_person_exercise_mapping.csv', index=False)


def main():
    dataset = OpenposeBodyJointsDataset(root_dir='',exercise_dir='openpose_data/',labels_dir='Annotations',
                                      target_frames=512, padding_size=1280)
    dataloader = DataLoader(dataset=dataset, batch_size=4)
    data_iter = iter(dataloader)
    
    video_paths, features, labels = next(data_iter)
    print(features.shape, labels.shape)
    #print(video_paths)
    
    #print(labels)
    
    # k.getKeypoints()
    # makeMapping()

    # makePersonIdVideoIdMapping()

    # df = pd.read_csv('video_person_exercise_mapping.csv')
    # print(df.head(20))


if __name__ == "__main__":
    main()
