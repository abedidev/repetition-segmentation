"""train TransRAC model """
from platform import node
import os
## if your data is .mp4 form, please use RepCountA_raw_Loader.py (slowly)
# from dataset.RepCountA_raw_Loader import MyData

## if your data is .npz form, please use RepCountA_Loader.py. It can speed up the training
# from dataset.RepCountA_Loader import MyData
# you can use 'tools.video2npz.py' to transform .mp4 to .npz
from TransRAC_v3 import TransferModel
from train_looping_cv import train_loop
from OpenposeFeatureLoader_v3 import OpenposeBodyJointsDataset

# CUDA environment
N_GPU = 1
device_ids = [i for i in range(N_GPU)]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# # # we pick out the fixed frames from raw video file, and we store them as .npz file
# # # we currently support 64 or 128 frames
# data root path
#root_path = 'D:\\WorkspaceD\\ProjR1_ExAs_P2\\Temp_Work\\TransRAC_Modified\\Gunin_Pratyush_Work\\Exercise-Agnostic-models\\Final_Raw_model\\'
#root_path = '/disk1/riddhi/Rep-Counting-Attention/'

#train_feature_path = 'E0_openpose'
#train_label_path = 'labels/0/all_annotations.csv'
#valid_video_dir = 'E0_openpose'
#valid_label_dir = 'labels/0/all_annotations_valid.csv'

# please make sure the pretrained model path is correct
# checkpoint = './pretrained/swin_tiny_patch244_window877_kinetics400_1k.pth'
# config = './configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py'

# TransRAC trained model checkpoint, we will upload soon.
#lastckpt = None

NUM_FRAME = 512
#NUM_FRAME = 64
PADDING_SIZE = 1280
# multi scales(list). we currently support 1,4,8 scale.
SCALES = [1, 4, 8]

#train_dataset = KinectBodyJointsDataset(root_dir='',exercise_dir='',labels_dir='Annotations')
dataset = OpenposeBodyJointsDataset(root_dir='',exercise_dir='openpose_data/',labels_dir='Annotations',
                                        target_frames=NUM_FRAME, padding_size=PADDING_SIZE)

#valid_dataset = MasterDataset(root_path, valid_video_dir, valid_label_dir, num_frame=NUM_FRAME)
#my_model = TransferModel(None, NUM_FRAME)
my_model = TransferModel(None, SCALES, PADDING_SIZE)

NUM_FOLDS = 5
NUM_EPOCHS = 81
#LR = 8e-6
LR = 8e-6
BATCH_SIZE = 8

#train_loop(NUM_EPOCHS, my_model, train_dataset, valid_set=None, train=True, valid=False,
#           batch_size=BATCH_SIZE, lr=LR, saveckpt=True, ckpt_name='ours', log_dir='ours', device_ids=device_ids,
#           lastckpt=lastckpt, mae_error=False)

train_loop(NUM_FOLDS, NUM_EPOCHS, my_model, dataset, batch_size=BATCH_SIZE, lr=LR)