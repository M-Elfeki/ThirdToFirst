import numpy as np, os, random, h5py
import matplotlib.pyplot as plt
os.environ['PYTHONHASHSEED'] = '1'

np.random.seed(1)
random.seed(1)
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from keras import backend as K

from keras.models import load_model


def return_rank_idx(row_val, diagonal_idx):
    row_sorted = np.argsort(row_val).tolist()
    return row_sorted.index(diagonal_idx)


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def evaluate_retrieval(data_paths, pre_trained_model_paths, output_similarity_path=None, single_size=500, is_side=True):
    all_top_k_ranks = []
    for i in range(len(data_paths)):
        all_top_k_ranks.append(compute_similarity(data_paths[i], pre_trained_model_paths[i], single_size, is_side))
    print len(all_top_k_ranks)
    if output_similarity_path is not None:
        for i in range(len(output_similarity_path)):
            h5f = h5py.File(output_similarity_path[i], 'w')

            h5f['top_k_rank'] = np.array(all_top_k_ranks[i])
            h5f.close()
    plot_rank_curves(all_top_k_ranks, pre_trained_model_paths)


def compute_similarity(data_path, pre_trained_model_path, single_size=500, is_side=True):
    h5f = h5py.File(data_path, 'r')
    pairs = np.array(h5f['pos_pairs_test']).astype('float')

    if 'Synthetic' in data_path:
        if is_side:
            pairs = pairs[:2]
        else:
            pairs = np.stack((pairs[0], pairs[2]), axis=0)

    pairs_list = []
    for i in range(0, pairs.shape[1], single_size):
        pairs_list.append(pairs[:, i:i+single_size])
    h5f.close()
    prediction_model = load_model(pre_trained_model_path, custom_objects={'contrastive_loss': contrastive_loss})
    print 'Model is Loaded!!'

    top_k_list = []
    for i in range(len(pairs_list)):
        top_k_list.append(compute_similarity_for_pairs(pairs_list[i], prediction_model))
    top_k_rank = np.array(top_k_list)

    return top_k_rank


def compute_similarity_for_pairs(pairs, prediction_model):

    rand_perm = np.random.permutation(pairs.shape[1])
    pairs = pairs[:, rand_perm]
    limit = pairs.shape[1]
    similarity = np.zeros((limit, limit))
    rand_similarity_arr = np.zeros((limit, limit))
    if pairs.shape[-1] != 150:
        v1 = np.rollaxis(pairs[0], 3, 1)
        v2 = np.rollaxis(pairs[1], 3, 1)
    else:
        v1 = pairs[0]
        v2 = pairs[1]
    ego_model = prediction_model.get_layer(index=-3)
    exo_model = prediction_model.get_layer(index=-2)

    ego_embedding = np.array(ego_model.predict(v1)).squeeze()
    exo_embedding = np.array(exo_model.predict(v2)).squeeze()
    for view_1_sample in range(limit):
        for view_2_sample in range(limit):
            similarity[view_1_sample][view_2_sample] = np.linalg.norm(ego_embedding[view_1_sample]-exo_embedding[view_2_sample])
            rand_similarity_arr[view_1_sample][view_2_sample] = np.random.random()

    cur_top_k_rank = np.zeros(limit)
    for sample_idx in range(limit):
        cur_top_k_rank[return_rank_idx(similarity[sample_idx], sample_idx)] += 1
    return cur_top_k_rank


def plot_rank_curves(all_top_k_rank, lbls):
    top_k_rank_list = []
    for cur_top_k_rank in all_top_k_rank:
        batches_number, limit = cur_top_k_rank.shape[0], cur_top_k_rank.shape[1]

        cur_top_k_rank = np.mean(cur_top_k_rank, axis=0) / float(limit)
        for i in range(len(cur_top_k_rank)):
            cur_top_k_rank[i] += cur_top_k_rank[i - 1]
        top_k_rank_list.append(cur_top_k_rank)

    top_k_counter_rand_list = []
    for i in range(batches_number):
        rand_similarity_arr = np.zeros((limit, limit))
        top_k_counter_rand = np.zeros(limit)
        for sample_idx in range(limit):
            rand_similarity_arr[sample_idx] = np.mean(np.random.random((limit, limit)), axis=1).squeeze()
            top_k_counter_rand[return_rank_idx(rand_similarity_arr[sample_idx], sample_idx)] += 1
        top_k_counter_rand_list.append(top_k_counter_rand)
    top_k_counter_rand = np.mean(top_k_counter_rand_list, axis=0) / float(limit)
    for i in range(1, len(top_k_counter_rand)):
        top_k_counter_rand[i] += top_k_counter_rand[i - 1]
    for top_k_rank_idx in range(len(top_k_rank_list)):
        top_k_counter = top_k_rank_list[top_k_rank_idx]
        mean_iou = np.array(top_k_counter).mean()
        plt.plot(np.linspace(0, 1, limit), top_k_counter,
                 label=lbls[top_k_rank_idx].split("/")[-1].split(".")[0].replace('Similarity_', '').replace('Model_', '').replace('_Side', '').replace('200_', '').replace('_1', '') + '_' + str(mean_iou))
    plt.plot(np.linspace(0, 1, limit), top_k_counter_rand,
             label='Random Rank_' + str(np.array(top_k_counter_rand).mean()))
    plt.legend()
    plt.xlabel('Top k- Rank')
    plt.ylabel('Frequency')
    plt.title('Domain Adaptation')
    plt.show()
