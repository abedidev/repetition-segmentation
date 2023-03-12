import math
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from label_norm import normalize_label


class KinectBodyJointsDataset(Dataset):
    def __init__(self,root_dir,exercise_dir,labels_dir, target_frames=128, padding_size=None):
        self.exercise_dir = exercise_dir
        self.labels_dir = labels_dir
        self.target_frames = target_frames
        self.num_frames = None
        self.padding_size = padding_size
        self.annotation_mapping = pd.read_csv(os.path.join(root_dir,'video_person_exercise_mapping.csv'))
        #self.makeMapping()
        #self.map = np.loadtxt(os.path.join(root_dir,'map.txt'), dtype='str', delimiter='\n')
        self.map = np.loadtxt(os.path.join(root_dir,'map.txt'), dtype='str')
        # print(self.map[2])

    def __len__(self):
        return len(self.map)

    # returns list of list of body joint coordinates for the video with given index in map.txt
    def __getitem__(self, index):
        # get the list of body joints for the video
        video_path = os.path.join(self.exercise_dir,self.map[index])
        features = self.getRawBodyJointFeatures(video_path)
        
        # get the person id and exercise number of the video
        personId, exerciseNum = video_path.split("/")[-4:-2]
        # print(personId,exerciseNum)
        exerciseNum = int(str(exerciseNum)[-1]) - 1

        # get the path of annotation file of this video
        labels_path = self.getLabelsPath(personId, exerciseNum)
        timestamps = self.get_timestamps(labels_path)
        labels = self.preprocess(
            self.num_frames, timestamps, self.num_frames)
               
        features, labels = self.pad_data(features, labels)
        
        t = torch.Tensor(features)
        labels = torch.Tensor(labels)
        # print(t.shape)   
        
        # print(labels)
        return video_path, t, labels
    
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
            

    def getRawBodyJointFeatures(self,video_path):
        # construct a list of landmarks
        lmlist = self.getLandmarks(video_path)
        self.num_frames = len(lmlist)
        # print(len(lmlist))

        # construct the feature list as [[x1,y1,z1,x2,y2,z2....][...][...]] where xi,yi,zi are the x,y,z coordinates of a body joint/sensor
        raw_features = []
        for frame in lmlist:
            frame_features = []
            for landmark in frame:
                x,y,z = landmark
                frame_features += [x,y,z]
            raw_features.append(frame_features)
        # print(raw_features)
        return raw_features

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

    def makeMapping(self):
        map_file = open('map.txt', 'w')
        kinect_features_dir = "OneDrive_1_12-9-2022"

        for patientType in os.listdir(kinect_features_dir):
            # patientType = CG or GPP
            patientTypeDir = os.path.join(kinect_features_dir, patientType)
            for patientDetailedType in os.listdir(patientTypeDir):
                patientDetailedTypeDir = os.path.join(patientTypeDir, patientDetailedType)
                for dirs in os.listdir(patientDetailedTypeDir):
                    # dirs = PATH_VAR/PersonId/
                    personId_dir = os.path.join(patientDetailedTypeDir, dirs)
                    if os.path.isdir(personId_dir):
                        for exerciseName in os.listdir(personId_dir):
                            if exerciseName in ["Es1", "Es2", "Es3", "Es4", "Es5"] and os.path.isdir(
                                    os.path.join(personId_dir, exerciseName)):
                                # print(f"ExerciseName = {exerciseName}")
                                # exercise_dirs = dirs/ExerciseName/Raw/
                                exercise_dir = os.path.join(personId_dir, exerciseName, 'Raw')
                                for fileName in os.listdir(exercise_dir):
                                    if str(fileName).startswith("JointPosition"):
                                        print(f'PersonId = {dirs}, exerciseNum = {exerciseName[-1]}')
                                        personId_df = self.annotation_mapping.loc[self.annotation_mapping['PersonID'] == dirs]
                                        print(personId_df)
                                        exNum = int(exerciseName[-1]) - 1
                                        video_df = personId_df.loc[personId_df['ExerciseNum'] == exNum]
                                        print(video_df)
                                        # add the video only if its annotation is available
                                        if len(video_df['AnnotationVideoNum'].values) == 1:
                                            map_file.write(os.path.join(exercise_dir, fileName) + '\n')

        map_file.close()

    def getLandmarks(self, video_body_joints_path):
        # take the join position input as CSV file
        # keypoint_csv = pd.read_csv(os.path.join(exercise_dir,fileName))
        # print(keypoint_csv.head(10))

        landmarks = []
        with open(video_body_joints_path) as f:
            for frame_num, frame_keypoints in enumerate(f.readlines()):

                keypointList = frame_keypoints.split(",")

                # landmarks.append([])
                x = []
                i = 0
                while i + 2 < len(keypointList):
                    x.append([float(keypointList[i]), float(keypointList[i + 1]), float(keypointList[i + 2])])
                    i += 4

                if len(x) == 25:
                    landmarks.append(x)

            # print(f' Number of keypoints in {frame_num} = {len(landmarks[-1])}')

        # l[dirs + "/" + exerciseName] = kpts
        # print(f'dict entry: {l[dirs + "/" + exerciseName]}')

        # print(f'x = {x}, y = {y}, z = {z}')
        f.close()

        # print(f'Landmarks for video {video_body_joints_path}')
        # print(landmarks)
        return landmarks

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
                

    def getLabelsPath(self, personId, exerciseNum):
        # print(personId, exerciseNum)
        # print(self.annotation_mapping.head(5))
        personId_df = self.annotation_mapping.loc[self.annotation_mapping['PersonID'] == personId]
        # print(personId_df)
        # for i in range(len(self.annotation_mapping)):
        #     if self.annotation_mapping.iloc[i,0] == personId and self.annotation_mapping.iloc[i,1] == exerciseNum:
        #         print(f'Video name = {self.annotation_mapping.iloc[i,2]}')
        video_df = personId_df.loc[personId_df['ExerciseNum'] == exerciseNum]
        videoName = video_df['AnnotationVideoNum'].values[0]

        video_path = os.path.join(self.labels_dir, 'E' + str(exerciseNum), 'csv_files', videoName + '.csv')

        # print(video_path)
        return video_path


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
    dataset = KinectBodyJointsDataset(root_dir='',exercise_dir='',labels_dir='Annotations',
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
