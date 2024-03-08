import pickle, os
import numpy as np
from tqdm import tqdm
from nsd_visuo_semantics.get_embeddings.embedding_models_zoo import get_embedding_model, get_embeddings
from nsd_visuo_semantics.decoding_analyses.decoding_utils import restore_nan_dims, sentences_zoo, interpolate_vectors, get_gcc_nearest_neighbour, load_gcc_embeddings


name_A = 'unique_sentence_people'  # must be a key in sentences_zoo (which is imported from decoding_utils)
name_B = 'unique_sentence_places'
name = f"{name_A}_to_{name_B}"

get_nn_sentences = True
get_predicted_voxels = True

EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
average_over = 'embeddings'  # embeddings or voxels
interpolate_steps = 100

fitted_models_dir = "/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/decoding_analyses/all-mpnet-base-v2_results_ROIfullbrain_encodingModel/fitted_models"
save_dir = '/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/decoding_analyses/all-mpnet-base-v2_results_ROIfullbrain_encodingModel/interpolate_project_embeddings/cache'
nsd_captions_path = f"/share/klab/adoerig/adoerig//nsd_visuo_semantics/src/nsd_visuo_semantics/get_embeddings/ms_coco_nsd_captions_test.pkl"
nsd_embeddings_path = f"/share/klab/adoerig/adoerig//nsd_visuo_semantics/results_dir/saved_embeddings/nsd_mpnet_mean_embeddings.pkl"

os.makedirs(save_dir, exist_ok=True)

sentences_A = sentences_zoo[name_A]
sentences_B = sentences_zoo[name_B]

embedding_model = get_embedding_model(EMBEDDING_MODEL_NAME)

loaded_captions, loaded_embeddings = load_gcc_embeddings()

embeds_A = get_embeddings(sentences_A, embedding_model, EMBEDDING_MODEL_NAME)
embeds_B = get_embeddings(sentences_B, embedding_model, EMBEDDING_MODEL_NAME)

if average_over == 'embeddings':
    embeds_A = np.mean(embeds_A, axis=0)[np.newaxis, :]
    embeds_B = np.mean(embeds_B, axis=0)[np.newaxis, :]

subs = [f'subj0{s}' for s in range(1,9)]

NN_sentences = []
for i, embeds in enumerate(tqdm(interpolate_vectors(embeds_A, embeds_B, interpolate_steps), desc="Interpolation Progress")):    
    
    if get_nn_sentences:
        sent, dist = get_gcc_nearest_neighbour(embeds, n_neighbours=1, 
                                        lookup_sentences=loaded_captions, 
                                        lookup_embeddings=loaded_embeddings,
                                        METRIC='cosine')
        NN_sentences.append(sent[0])
    
    if get_predicted_voxels:
        pred_voxels_all = []
        for subj in subs:

            model_save_path = f"{fitted_models_dir}/{subj}_fittedFracridgeEncodingModel_fullbrain.pkl"
            with open(model_save_path, "rb") as f:
                fitted_fracridge = pickle.load(f)

            pred_voxelsA = fitted_fracridge.predict(embeds).squeeze()

            nan_idx_to_restore = np.load(f"{fitted_models_dir}/{subj}_NanIdxToRestore.npy")
            pred_voxels = restore_nan_dims(pred_voxelsA, nan_idx_to_restore, axis=0)

            np.save(f"{save_dir}/{subj}_pred_voxels_{name}_interp{i}.npy", pred_voxels)

            pred_voxels_all.append(pred_voxels)

    if get_predicted_voxels:
        # average all_voxels, and save
        pred_voxels_all = np.mean(pred_voxels_all, axis=0)

        np.save(f"{save_dir}/subjAvg_pred_voxels_{name}_interp{i}.npy", pred_voxels_all)


if get_nn_sentences:
    [print(s[0]+'\n') for s in NN_sentences]
    # Write strings to the text file
    with open(f'{save_dir}/nn_sentences.txt', 'w') as file:
        for string in NN_sentences:
            file.write(string[0] + '\n')
