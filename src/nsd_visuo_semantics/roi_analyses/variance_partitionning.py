import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from nsd_visuo_semantics.utils.nsd_get_data_light import get_model_rdms
# import statsmodels.api as sm
from sklearn.linear_model import LinearRegression




def variance_partitioning(y, predictors_features, predictors_names, zscore=True, return_np=False, verbose=False):
    '''y: target variable (brain RDM)
    predictors_names: list of predictors names (one per model you want to do variance decomposition on)
    predictors_features: array of predictors features (one per model you want to do variance decomposition on)'''

    if len(y.shape) == 1:
        y = y[:, np.newaxis]

    for f in predictors_features:
        assert f.shape[0] == y.shape[0], \
        f"Predictors features must have the same length as the target variable, got {f.shape[0]} and {y.shape[0]}"
        if len(f.shape) == 1:
            f = f[:, np.newaxis]

    # Create all possible combinations of predictors
    all_combinations = []
    for r in range(1, len(predictors_names) + 1):
        all_combinations.extend(combinations(predictors_names, r))

    if zscore:
        from scipy.stats import zscore
        y = zscore(y, axis=0)
        predictors_features = [zscore(f, axis=0) for f in predictors_features]
        if all(np.isnan(y)):
            if verbose:
                print("All values in y are NaNs after zscoring. This is likely because y is constant across samples. "\
                    "We will skip model fitting and return all scores as NaNs.")
            if return_np:
                return np.array([np.nan for c in all_combinations]), all_combinations
            else:
                return {c: np.nan for c in all_combinations}
        for f in predictors_features:
            if all(np.isnan(f)):
                if verbose:
                    print("All values in one or more predictors are NaNs after zscoring. "\
                        "This is likely because the predictor(s) is constant across samples. "\
                        "We will skip model fitting and return all scores as NaNs.")
                if return_np:
                    return np.array([np.nan for c in all_combinations]), all_combinations
                else:
                    return {c: np.nan for c in all_combinations}

    # Fit models for each combination of predictors
    models = {}
    var_components = {}
    for combo in all_combinations:
        predictors_idx = [predictors_names.index(p) for p in combo]
        X = np.column_stack([predictors_features[idx] for idx in predictors_idx])
        model = LinearRegression(fit_intercept=False).fit(X, y)
        models[combo] = model
        var_components[combo] = model.score(X,y)

    # Print variance components
    if verbose:
        for combo, var in var_components.items():
            print(f"Rsquared {combo}: {var}")

    if return_np:
        return np.array([var_components[c] for c in all_combinations]), all_combinations
    else:
        return var_components
    

def combination_scores_to_unique_var(scores, normalize_areas=True):

    if len(scores) == 3:
        A, B, A_B = scores
        Ab = A_B - B
        aB = A_B - A
        AB = A + B - A_B
        if normalize_areas:
            Ab /= A_B
            aB /= A_B
            AB /= A_B
        return Ab, aB, AB
    
    elif len(scores) == 7:
        A, B, C, A_B, A_C, B_C, A_B_C = scores
        Abc = A_B_C - B_C
        aBc = A_B_C - A_C
        abC = A_B_C - A_B
        ABc = A_C + B_C - A_B_C - C
        AbC = A_B + B_C - A_B_C - B
        aBC = A_C + A_B - A_B_C - A
        ABC = A + B + C - A_B - A_C - B_C + A_B_C
        if normalize_areas:
            Abc /= A_B_C
            aBc /= A_B_C
            ABc /= A_B_C
            abC /= A_B_C
            AbC /= A_B_C
            aBC /= A_B_C
            ABC /= A_B_C
        return Abc, aBc, ABc, abC, AbC, aBC, ABC


if __name__ == "__main__":

    MODEL_NAMES = [
    # 'all-mpnet-base-v2_nsd_special100_cocoCaptions',
    # 'all-mpnet-base-v2_nouns_nsd_special100_cocoCaptions',
    # 'all-mpnet-base-v2_verbs_nsd_special100_cocoCaptions',
    'mpnet',
    'mpnet_nouns',
    'mpnet_verbs',
    # 'noisy_brain',  # can be used for sanity checks, etc
    ]

    n_subjects = 8
    subs = [f"subj0{x+1}" for x in range(n_subjects)]
    rdm_distance = 'correlation'
    remove_shared_515 = False
    which_rois = 'streams'

    normalize_areas = True  # normalize areas of venn diagram to 1
    zscore = True

    # set up directories
    base_save_dir = "/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir"
    models_dir = os.path.join(base_save_dir,f'serialised_models{"_noShared515" if remove_shared_515 else ""}_{rdm_distance}')
    roi_analyses_dir = os.path.join(base_save_dir, "roi_analyses")
    os.makedirs(roi_analyses_dir, exist_ok=True)
    results_dir = os.path.join(roi_analyses_dir, f"{which_rois}_roi_results_{rdm_distance}")
    os.makedirs(results_dir, exist_ok=True)
    subj_roi_rdms_path = os.path.join(results_dir, "subj_roi_rdms")
    os.makedirs(subj_roi_rdms_path, exist_ok=True)

    var_components_save_file = os.path.join(f"./variance_components_{rdm_distance}_zscore{'special100' if 'special100' in MODEL_NAMES[0] else ''}.npy")
    if os.path.exists(var_components_save_file):
        all_var_components = np.load(var_components_save_file, allow_pickle=True).item()
    else:
        all_var_components = {}

    for subj in subs:

        if subj not in all_var_components.keys():
            all_var_components[subj] = {}

        fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(2*6, 4*6))
        ax[0][0].axis('off')  # we won't draw here.

        for m, mask_name in enumerate(['early', 'midventral', 'ventral', 'midlateral', 'lateral', 'midparietal', 'parietal']):

            if 'special100' in MODEL_NAMES[0]:
                brain_rdm_file = os.path.join(subj_roi_rdms_path, f"{subj}_{mask_name}_100rdm_{rdm_distance}.npy")
            else:
                brain_rdm_file = os.path.join(subj_roi_rdms_path, f"{subj}_{mask_name}_fullrdm_{rdm_distance}.npy")
            brain_rdm = np.load(brain_rdm_file)

            all_model_rdms, all_model_names = [], []
            for MODEL_NAME in MODEL_NAMES:
                if MODEL_NAME == 'noisy_brain':
                    # can be used for sanity checks, etc
                    all_model_rdms.append(brain_rdm+np.random.normal(0, 0.01, brain_rdm.shape))
                    all_model_names.append('noisy_brain')
                else:
                    model_rdms, model_names = get_model_rdms(f"{models_dir}/{MODEL_NAME}", subj, filt=MODEL_NAME)
                    all_model_rdms.append(model_rdms[0])
                    all_model_names.append(model_names[0].replace('_nsd_special100_cocoCaptions', '').replace('all-mpnet-base-v2', 'mpnet').replace('cutoffDist0.7_', ''))

            # from nsd_visuo_semantics.utils.utils import corr_rdms
            # print(corr_rdms(all_model_rdms[0][None,:], brain_rdm[None,:]))
            # import pdb; pdb.set_trace()

            if mask_name in all_var_components[subj].keys():
                print(f"Skipping {mask_name} for {subj} because it was already done")
            else:
                var_components = variance_partitioning(brain_rdm, all_model_rdms, all_model_names, zscore)
                all_var_components[subj][mask_name] = var_components

            # Plot Venn diagram
            # Create all possible combinations of predictors
            k = []
            for r in range(1, len(all_model_names) + 1):
                k.extend(combinations(all_model_names, r))
            # k = list(var_components.keys())
            labels = [' & '.join(combo) for combo in k]

            # helper function to round numbers in venn diagram for readability
            def decimals(n):
                return np.round(n, decimals=4)

            if len(all_model_names) == 2:
                from matplotlib_venn import venn2
                A = all_var_components[subj][mask_name][k[0]]  # score for 1st model
                B = all_var_components[subj][mask_name][k[1]]  # score for 2nd model
                A_B = all_var_components[subj][mask_name][k[2]]  # score of combined model
                Ab, aB, AB = combination_scores_to_unique_var([A, B, A_B])
                venn2(subsets=[Ab, aB, AB], set_labels=labels, subset_label_formatter=decimals, ax=ax[(m+1)//2][(m+1)%2])
                ax[(m+1)//2][(m+1)%2].set_title(mask_name, fontsize=16, fontweight='bold')

            elif len(all_model_names) == 3:
                from matplotlib_venn import venn3
                A = all_var_components[subj][mask_name][k[0]]  # score for 1st model
                B = all_var_components[subj][mask_name][k[1]]  # score for 2nd model
                C = all_var_components[subj][mask_name][k[2]]  # score for 3rd model
                A_B = all_var_components[subj][mask_name][k[3]]
                A_C = all_var_components[subj][mask_name][k[4]]
                B_C = all_var_components[subj][mask_name][k[5]] 
                A_B_C = all_var_components[subj][mask_name][k[6]]
                Abc, aBc, ABc, abC, AbC, aBC, ABC = combination_scores_to_unique_var([A, B, C, A_B, A_C, B_C, A_B_C])
                venn3(subsets=[Abc, aBc, ABc, abC, AbC, aBC, ABC], set_labels=labels, subset_label_formatter=decimals, ax=ax[(m+1)//2][(m+1)%2])
                ax[(m+1)//2][(m+1)%2].set_title(mask_name, fontsize=16, fontweight='bold')

        np.save(var_components_save_file, all_var_components)

        plt.savefig(f'./venn_diagram_{"spec100CocoCap_" if "special100_cocoCaptions" in MODEL_NAMES[0] else ""}{subj}{"_zscore" if zscore else ""}.png')
        plt.tight_layout()
        plt.close()

    # Plot average variance components across subjects
    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(2*6, 4*6))
    ax[0][0].axis('off')  # we won't draw here.
    for m, mask_name in enumerate(['early', 'midventral', 'ventral', 'midlateral', 'lateral', 'midparietal', 'parietal']):
        if len(all_model_names) == 2:
            raise NotImplementedError
        elif len(all_model_names) == 3:
            avg_A = np.mean([all_var_components[s][mask_name][k[0]] for s in subs])  # score for 1st model
            std_A = np.std([all_var_components[s][mask_name][k[0]] for s in subs])/np.sqrt(len(subs))  # score for 1st model
            avg_B = np.mean([all_var_components[s][mask_name][k[1]] for s in subs])
            std_B = np.std([all_var_components[s][mask_name][k[1]] for s in subs])/np.sqrt(len(subs))
            avg_C = np.mean([all_var_components[s][mask_name][k[2]] for s in subs])
            std_C = np.std([all_var_components[s][mask_name][k[2]] for s in subs])/np.sqrt(len(subs))
            avg_A_B = np.mean([all_var_components[s][mask_name][k[3]] for s in subs])
            std_A_B = np.std([all_var_components[s][mask_name][k[3]] for s in subs])/np.sqrt(len(subs))
            avg_A_C = np.mean([all_var_components[s][mask_name][k[4]] for s in subs])
            std_A_C = np.std([all_var_components[s][mask_name][k[4]] for s in subs])/np.sqrt(len(subs))
            avg_B_C = np.mean([all_var_components[s][mask_name][k[5]] for s in subs])
            std_B_C = np.std([all_var_components[s][mask_name][k[5]] for s in subs])/np.sqrt(len(subs))
            avg_A_B_C = np.mean([all_var_components[s][mask_name][k[6]] for s in subs])
            std_A_B_C = np.std([all_var_components[s][mask_name][k[6]] for s in subs])/np.sqrt(len(subs))
            Abc, aBc, ABc, abC, AbC, aBC, ABC = combination_scores_to_unique_var([avg_A, avg_B, avg_C, avg_A_B, avg_A_C, avg_B_C, avg_A_B_C])
        venn3(subsets=[Abc, aBc, ABc, abC, AbC, aBC, ABC], set_labels=labels, subset_label_formatter=decimals, ax=ax[(m+1)//2][(m+1)%2])
        ax[(m+1)//2][(m+1)%2].set_title(mask_name, fontsize=16, fontweight='bold')

    plt.savefig(f'./venn_diagram_{"spec100CocoCap_" if "special100_cocoCaptions" in MODEL_NAMES[0] else ""}{subj}{"_zscore" if zscore else ""}_subjAvg.png')
    plt.tight_layout()
    plt.close()
