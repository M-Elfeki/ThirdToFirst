import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from RetrievalNetwork import start_training
from EvaluateRetrieval import evaluate_retrieval

vs = 'Side'         # 'Top' OR 'Side'
data_type = 'OF'    # 'RGB_Frames' OR 'OF'

# If top==> vs =[3,2]

models_path = '/home/elfeki/Desktop/'
real_data = '/home/elfeki/Workspace/Generation_Datasets/Real_Dataset/'+data_type+'/'+vs+'/'
synthetic_data = '/home/elfeki/Workspace/Generation_Datasets/Synthetic_Dataset/'+data_type+'/'

real_testing_data = real_data + 'Real_'+vs+'_TestingData.h5'
synthetic_testing_data = synthetic_data + 'Synthetic_TestingData.h5'

synthetic_model = models_path + 'Model_Synthetic_'+vs+'_'+data_type+'.h5'
real_model = models_path + 'Model_Real_'+vs+'_'+data_type+'.h5'
real_model_da = models_path + 'Model_Real_DA_'+vs+'_'+data_type+'.h5'
baseline_similarity = models_path + 'Similarity_TrainedSynthetic_TestedReal_'+vs+'_'+data_type+'.h5'

samples_num_synth = 240000
samples_num_real = 100000
if data_type == 'OF':
    imgs_size = (2, 100, 150)
else:
    imgs_size = (3, 224, 224)


start_training(data_path=synthetic_data, output_model_path=synthetic_model,
               num_samples=samples_num_synth, images_size=imgs_size)

start_training(data_path=real_data, output_model_path=real_model,
              num_samples=samples_num_real, images_size=imgs_size)

start_training(data_path=real_data, output_model_path=real_model_da,
               pre_trained_model_path=synthetic_model,
               num_samples=samples_num_real, images_size=imgs_size)



evaluate_retrieval(data_paths=[synthetic_testing_data, real_testing_data, real_testing_data, real_testing_data],
                   pre_trained_model_paths=[synthetic_model, synthetic_model, real_model, real_model_da],
                   output_similarity_path=[synthetic_model.replace('Model_', 'Similarity'), baseline_similarity,
                                           real_model.replace('Model_', 'Similarity_'), real_model_da.replace('Model_', 'Similarity_')],
                   single_size=500, is_side=(vs == 'Side'))
