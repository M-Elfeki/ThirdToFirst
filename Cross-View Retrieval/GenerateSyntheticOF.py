import numpy as np, os, re, scipy.io as sio, scipy.ndimage as ndimage, sys, random, cv2, h5py
np.random.seed(100)
random.seed(100)

writing_ds_data = '/home/elfeki/Workspace/Generation_Datasets/Synthetic_Dataset/OF/'

flow_len = 30
gaussian_weights = ndimage.gaussian_filter1d(np.float_([0]*(flow_len/2)+[1]+[0]*((flow_len/2)-1)), 1)
counter = 0

def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def return_gaussian_mean(input_list):
    mean_image = np.zeros_like(input_list[0])
    for i in range(len(input_list)):
        mean_image += gaussian_weights[i] * input_list[i]
    return mean_image


# actions_list = ['Straffing', 'Strafing', 'Run', 'Walk', 'Jump', 'Crouch', 'Sprinting']
actions_list = ['Strafing', 'Run', 'Walk', 'Jump', 'Crouch', 'Sprinting']
# Camera-1: EGO, Camera-2:Side, Camera-3:Top
counter_list = [0] * len(actions_list)


def return_data_for_sample(sample_dir):
    cur_sample = [[], [], [], [], []]  # <Camera_1, Camera_2, Camera_3, Set, FrameNum>
    min_length = sys.maxint
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
        opt_x, opt_y = [], []
        for cur_frame_idx in range(0, min_length, 2):
            cur_x = np.array(sio.loadmat(str(sample_dir + cur_camera + '/' + frames[cur_frame_idx]).replace('.jpg', '.mat').replace('.png', '.mat'))['x'])
            cur_y = np.array(sio.loadmat(str(sample_dir + cur_camera + '/' + frames[cur_frame_idx+1]).replace('.jpg', '.mat').replace('.png', '.mat'))['y'])

            opt_x.append(cur_x.astype(np.float64))
            opt_y.append(cur_y.astype(np.float64))
        for cur_frame_idx in range((flow_len / 2), len(opt_x) - (flow_len / 2)):
            avg_x = return_gaussian_mean(opt_x[cur_frame_idx - (flow_len / 2):cur_frame_idx + (flow_len / 2)])
            avg_y = return_gaussian_mean(opt_y[cur_frame_idx - (flow_len / 2):cur_frame_idx + (flow_len / 2)])
            cur_sample[temp_idx].append(np.array([avg_x, avg_y]))
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
            if len(cam_1) > 0 and len(cam_1) == len(cam_2) == len(cam_3):
                if is_training == 1:
                    write_list_to_folder(writing_ds_data + 'Training/', cam_1, cam_2, cam_3, seq_name, frame_num, environment_num)
                elif is_training == 2:
                    write_list_to_folder(writing_ds_data + 'Validation/', cam_1, cam_2, cam_3, seq_name, frame_num, environment_num)
                else:
                    write_list_to_folder(writing_ds_data + 'Testing/', cam_1, cam_2, cam_3, seq_name, frame_num, environment_num)


def write_list_to_folder(folder_dir, cam_1, cam_2, cam_3, seq_name, frame_num, environment_num):
    global counter
    for k in range(len(cam_1)):
        sio.savemat(str(folder_dir + 'Camera_2/' + str(counter) + '_' + str(environment_num) + '_' + seq_name[k] + '_' + str(frame_num[k]) + '.mat'), mdict={'opt_flow': cam_1[k]})
        sio.savemat(str(folder_dir + 'Camera_1/' + str(counter) + '_' + str(environment_num) + '_' + seq_name[k] + '_' + str(frame_num[k]) + '.mat'), mdict={'opt_flow': cam_2[k]})
        sio.savemat(str(folder_dir + 'Camera_3/' + str(counter) + '_' + str(environment_num) + '_' + seq_name[k] + '_' + str(frame_num[k]) + '.mat'), mdict={'opt_flow': cam_3[k]})
        # print counter,
        counter += 1


def generate_pos_pairs_for_dataset(optical_flow_dir, environment_num):
    samples_list = os.listdir(optical_flow_dir)
    random.shuffle(samples_list)
    for action_name in actions_list:
        return_data_for_action(action_name, optical_flow_dir, samples_list, environment_num)
    print environment_num


def generate_OF_Synth():
    print writing_ds_data
    first_ds_data = '/media/elfeki/Mohamed_Backup/View-Generation/Generation_Datasets/Synthetic_Dataset/Synth_Dataset_1/Optical_Flow_lk/'
    second_ds_data = '/media/elfeki/Mohamed_Backup/View-Generation/Generation_Datasets/Synthetic_Dataset/Synth_Dataset_2/Optical_Flow_lk/'
    third_ds_data = '/media/elfeki/Mohamed_Backup/View-Generation/Generation_Datasets/Synthetic_Dataset/Synth_Dataset_3/Optical_Flow_lk/'
    forth_ds_data = '/media/elfeki/Mohamed_Backup/View-Generation/Generation_Datasets/Synthetic_Dataset/Synth_Dataset_4/Optical_Flow_lk/'
    generate_pos_pairs_for_dataset(first_ds_data, 1)
    generate_pos_pairs_for_dataset(second_ds_data, 2)
    generate_pos_pairs_for_dataset(third_ds_data, 3)
    generate_pos_pairs_for_dataset(forth_ds_data, 4)


def generate_Testing_Data():
    np.random.seed(1)
    random.seed(1)
    files_dir = writing_ds_data + 'Testing/'

    opt_files = [[], [], []]
    f = sorted(os.listdir(files_dir+'Camera_1/'))
    random.shuffle(f)
    f = f[:2000]
    for cur_camera_idx in range(len(sorted(os.listdir(files_dir)))):
        cur_camera = sorted(os.listdir(files_dir))[cur_camera_idx]
        for file_name in f:
            img = np.array(sio.loadmat(files_dir + cur_camera + '/' + file_name)['opt_flow'])
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


generate_OF_Synth()
generate_Testing_Data()
