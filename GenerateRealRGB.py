import h5py, numpy as np, os, re, sys, cv2, random
import scipy.io as sio
from scipy.misc import imresize, imsave
from PIL import Image

random.seed(1)
np.random.seed(1)

presp = 'Side'#'Side'

data_dir = '/media/elfeki/Mohamed_Backup/View-Generation/Generation_Datasets/Real_Dataset/Frames/'+presp+'/'
output_path = '/home/elfeki/Workspace/Generation_Datasets/Real_Dataset/RGB_Frames/'+presp+'/'
annotations_path = '/home/elfeki/Workspace/Generation_Datasets/Real_Dataset/Annotations/'
actions_list = ['Running', 'Jumping', 'Walking', 'Jogging', 'waving', 'clapping', 'boxing']

data_size = (224, 224)
counter = 0
samples_list = os.listdir(data_dir)
random.shuffle(samples_list)


def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def compress_annotations(ann):
    result = []
    cur_action = ann[0]
    cur_action_start = 0
    for i in range(1, len(ann)):
        if cur_action != ann[i]:
            result.append(cur_action + '_' + str(cur_action_start*4) + '_' + str(i*4))
            cur_action = ann[i]
            cur_action_start = i
    result.append(cur_action + '_' + str(cur_action_start*4) + '_' + str(len(ann)*4))
    return result


def return_data_for_set_for_action(sample_dir, action_type):
    cur_sample = [[], [], [], [], []] # <Camera_1, Camera_2, Set, Action, FrameNum>
    min_length = sys.maxint
    for cur_camera_idx in range(2):
        cur_camera = sorted(os.listdir(sample_dir))[cur_camera_idx]
        if len(os.listdir(sample_dir + cur_camera)) < min_length:
            min_length = len(os.listdir(sample_dir + cur_camera))
    set_path = sample_dir[sample_dir.find('Set'):]
    annotations = np.array(sio.loadmat(annotations_path + set_path + 'Ego_0/Ego_0_fc6.mat')['features']['action']).squeeze()
    for i in range(len(annotations)):
        annotations[i] = str(np.char.encode(annotations[i][0], encoding="ascii"))
    annotations = compress_annotations(annotations)
    for cur_camera_idx in range(2):
        cur_camera = sorted(os.listdir(sample_dir))[cur_camera_idx]
        frames = sorted(os.listdir(sample_dir + cur_camera), key=numerical_sort)
        for cur_action_range in annotations:
            arr = cur_action_range.split('_')
            cur_action = arr[0]
            start, end = int(arr[1]), int(arr[2])
            if cur_action.lower() != action_type.lower():
                continue
            for cur_frame_idx in range(start, end):
                cur_sample[cur_camera_idx].append(str(sample_dir + cur_camera + '/' + frames[cur_frame_idx-1]))
                if cur_camera_idx > 0:
                    cur_sample[2].append(set_path.replace('/', ''))
                    cur_sample[3].append(action_type)
                    cur_sample[4].append(cur_frame_idx)

    return cur_sample


def return_data_for_action(cur_action):
    for cur_sample in samples_list:
        is_training = 1 # training samples
        rand_chance = random.random()
        if 0.5 <= rand_chance <= 0.75:
            is_training = 2 # validation samples
        elif rand_chance > 0.75:
            is_training = 3 # testing samples

        cam_1, cam_2, set_num, action_name, frame_num = return_data_for_set_for_action(data_dir+cur_sample+'/', cur_action)

        if len(cam_1) > 0:
            if is_training == 1:
                write_list_to_folder(cam_1, cam_2, set_num, action_name, frame_num, output_path + 'Training/')
            elif is_training == 2:
                write_list_to_folder(cam_1, cam_2, set_num, action_name, frame_num, output_path + 'Validation/')
            else:
                write_list_to_folder(cam_1, cam_2, set_num, action_name, frame_num, output_path + 'Testing/')


def write_list_to_folder(cam_1, cam_2, set_num, action_name, frame_num, folder_dir):
    global counter
    for k in range(len(cam_1)):
        cur_img_cam1 = imresize(np.array(Image.open(cam_1[k])), data_size)
        imsave(folder_dir + 'Camera_1/' + str(counter) + '_' + str(set_num[k]) + '_' + action_name[k] + '_' + str(frame_num[k]) + '.jpg', cur_img_cam1)
        cur_img_cam2 = imresize(np.array(Image.open(cam_2[k])), data_size)
        imsave(folder_dir + 'Camera_2/' + str(counter) + '_' + str(set_num[k]) + '_' + action_name[k] + '_' + str(frame_num[k]) + '.jpg', cur_img_cam2)
        print counter,
        counter += 1


def generate_Real_RGB():
    print output_path
    for action_name in actions_list:
        return_data_for_action(action_name)


def generate_Testing_Data():
    np.random.seed(1)
    random.seed(1)
    files_dir = output_path + 'Testing/'

    opt_files = [[], []]
    f = sorted(os.listdir(files_dir+'Camera_1/'))
    random.shuffle(f)
    f = f[:2000]
    for cur_camera_idx in range(len(sorted(os.listdir(files_dir)))):
        cur_camera = sorted(os.listdir(files_dir))[cur_camera_idx]
        if cur_camera == 'Camera_3':
            continue
        for file_name in f:
            img = np.array(Image.open(files_dir+cur_camera+'/'+file_name))
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            opt_files[cur_camera_idx].append(img)

    cam_1 = np.array(opt_files[0])
    cam_2 = np.array(opt_files[1])
    print cam_1.shape
    rand_perm = np.random.permutation(len(cam_1))
    cam_1 = cam_1[rand_perm]
    cam_2 = cam_2[rand_perm]

    a = np.array([cam_1, cam_2])[:, :2000]
    print a.shape

    h5f = h5py.File(output_path+'Real_'+presp+'_TestingData.h5', 'w')
    h5f['pos_pairs_test'] = a
    h5f.close()

# generate_Real_RGB()
generate_Testing_Data()
