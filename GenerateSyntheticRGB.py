import numpy as np, os, re, sys, random, cv2, h5py
from PIL import Image
from scipy.misc import imresize, imsave

np.random.seed(1)
random.seed(1)

data_size = (224,224)
# actions_list = ['Straffing', 'Strafing', 'Run', 'Walk', 'Jump', 'Crouch', 'Sprinting']
actions_list = ['Strafing', 'Run', 'Walk', 'Jump', 'Crouch', 'Sprinting']
writing_ds_data = '/home/elfeki/Workspace/Generation_Datasets/Synthetic_Dataset/RGB_Frames/'

# Camera-1: EGO, Camera-2:Side, Camera-3:Top

counter = 0


def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def return_data_for_sample(sample_dir):
    cur_sample = [[], [], [], [], []] # <Camera_1, Camera_2, Camera_3, Set, FrameNum>
    min_length = sys.maxint
    if len(os.listdir(sample_dir)) == 0:
        return cur_sample
    for cur_camera_idx in range(3):
        cur_camera = sorted(os.listdir(sample_dir))[cur_camera_idx]
        if len(os.listdir(sample_dir + cur_camera)) < min_length:
            min_length = len(os.listdir(sample_dir + cur_camera))
    for cur_camera_idx in range(3):
        cur_camera = sorted(os.listdir(sample_dir))[cur_camera_idx]
        if 'Synth_Dataset_3' in sample_dir and cur_camera_idx < 2:
            temp_idx = 1 - cur_camera_idx
        else:
            temp_idx = cur_camera_idx
        frames = sorted(os.listdir(sample_dir + cur_camera), key=numerical_sort)
        for cur_frame_idx in range(0, min_length):
            cur_sample[temp_idx].append(str(sample_dir + cur_camera + '/' + frames[cur_frame_idx]))
            if temp_idx == 1:
                cur_sample[3].append(sample_dir.split('/')[-2])
                cur_sample[4].append(cur_frame_idx)

    return cur_sample


def return_data_for_action(action_name, optical_flow_dir, samples_list, environment_num):

    for cur_sample in samples_list:
        if action_name.lower() in cur_sample.lower():
            is_training = 1 # training samples
            rand_chance = random.random()
            if 0.5 <= rand_chance <= 0.75:
                is_training = 2 # validation samples
            elif rand_chance > 0.75:
                is_training = 3 # testing samples

            cam_1, cam_2, cam_3, seq_name, frame_num = return_data_for_sample(optical_flow_dir+cur_sample+'/')
            if len(cam_1) > 0:
                if is_training == 1:
                    write_list_to_folder(writing_ds_data + 'Training/', cam_1, cam_2, cam_3, seq_name, frame_num, environment_num)
                elif is_training == 2:
                    write_list_to_folder(writing_ds_data + 'Validation/', cam_1, cam_2, cam_3, seq_name, frame_num, environment_num)
                else:
                    write_list_to_folder(writing_ds_data + 'Testing/', cam_1, cam_2, cam_3, seq_name, frame_num, environment_num)


def write_list_to_folder(folder_dir, cam_1, cam_2, cam_3, seq_name, frame_num, environment_num):
    global counter
    for k in range(len(cam_1)):
        cur_img_cam1 = imresize(np.array(Image.open(cam_1[k])), data_size)
        imsave(folder_dir + 'Camera_2/' + str(counter) + '_' + str(environment_num) + '_' + seq_name[k] + '_' + str(frame_num[k]) + '.jpg', cur_img_cam1)   # Side
        cur_img_cam2 = imresize(np.array(Image.open(cam_2[k])), data_size)
        imsave(folder_dir + 'Camera_1/' + str(counter) + '_' + str(environment_num) + '_' + seq_name[k] + '_' + str(frame_num[k]) + '.jpg', cur_img_cam2)   # EGO
        cur_img_cam3 = imresize(np.array(Image.open(cam_3[k])), data_size)
        imsave(folder_dir + 'Camera_3/' + str(counter) + '_' + str(environment_num) + '_' + seq_name[k] + '_' + str(frame_num[k]) + '.jpg', cur_img_cam3)  # Top
        # print counter,
        counter += 1


def generate_pos_pairs_for_dataset(optical_flow_dir, environment_num):
    samples_list = os.listdir(optical_flow_dir)
    random.shuffle(samples_list)
    for action_name in actions_list:
        return_data_for_action(action_name, optical_flow_dir, samples_list, environment_num)
    print environment_num


def generate_RGB_Synth():
    first_ds_data = '/media/elfeki/Mohamed_Backup/View-Generation/Generation_Datasets/Synthetic_Dataset/Synth_Dataset_1/Frames/'
    second_ds_data = '/media/elfeki/Mohamed_Backup/View-Generation/Generation_Datasets/Synthetic_Dataset/Synth_Dataset_2/Frames/'
    third_ds_data = '/media/elfeki/Mohamed_Backup/View-Generation/Generation_Datasets/Synthetic_Dataset/Synth_Dataset_3/Frames/'
    forth_ds_data = '/media/elfeki/Mohamed_Backup/View-Generation/Generation_Datasets/Synthetic_Dataset/Synth_Dataset_4/Frames/'
    print writing_ds_data
    generate_pos_pairs_for_dataset(first_ds_data, 1)
    generate_pos_pairs_for_dataset(second_ds_data, 2)
    generate_pos_pairs_for_dataset(third_ds_data, 3)
    generate_pos_pairs_for_dataset(forth_ds_data, 4)


def generate_Testing_Data():
    np.random.seed(1)
    random.seed(1)
    files_dir = writing_ds_data+'/Testing/'

    opt_files = [[], [], []]
    f = sorted(os.listdir(files_dir+'Camera_1/'))
    random.shuffle(f)
    f = f[:2000]
    for cur_camera_idx in range(len(sorted(os.listdir(files_dir)))):
        cur_camera = sorted(os.listdir(files_dir))[cur_camera_idx]
        for file_name in f:
            img = np.array(Image.open(files_dir+cur_camera+'/'+file_name))
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            opt_files[cur_camera_idx].append(img)

    cam_1 = np.array(opt_files[0])
    cam_2 = np.array(opt_files[1])
    cam_3 = np.array(opt_files[2])
    print cam_1.shape
    rand_perm = np.random.permutation(len(cam_1))
    cam_1 = cam_1[rand_perm]
    cam_2 = cam_2[rand_perm]
    cam_3 = cam_3[rand_perm]

    a = np.array([cam_1, cam_2, cam_3])[:, :2000]
    print a.shape

    h5f = h5py.File(writing_ds_data+'Synthetic_TestingData.h5', 'w')
    h5f['pos_pairs_test'] = a
    h5f.close()


generate_RGB_Synth()
generate_Testing_Data()
