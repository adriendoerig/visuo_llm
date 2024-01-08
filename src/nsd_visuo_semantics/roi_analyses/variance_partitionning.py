import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from nsd_visuo_semantics.utils.nsd_get_data_light import get_model_rdms
# import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

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


def variance_partitioning(y, predictors_features, predictors_names):
    '''y: target variable
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

    # Fit models for each combination of predictors
    models = {}
    var_components = {}
    for combo in all_combinations:
        predictors_idx = [predictors_names.index(p) for p in combo]
        X = np.column_stack([predictors_features[idx] for idx in predictors_idx])
        # X = sm.add_constant(X)
        # model = sm.OLS(y, X).fit()
        model = LinearRegression(fit_intercept=True).fit(X, y)
        models[combo] = model
        # var_components = {combo: model.rsquared for combo, model in models.items()}
        var_components[combo] = model.score(X,y)

    # Print variance components
    for combo, var in var_components.items():
        print(f"Rsquared {combo}: {var}")

    return var_components


# set up directories
base_save_dir = "/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir"
models_dir = os.path.join(base_save_dir,f'serialised_models{"_noShared515" if remove_shared_515 else ""}_{rdm_distance}')
roi_analyses_dir = os.path.join(base_save_dir, "roi_analyses")
os.makedirs(roi_analyses_dir, exist_ok=True)
results_dir = os.path.join(roi_analyses_dir, f"{which_rois}_roi_results_{rdm_distance}")
os.makedirs(results_dir, exist_ok=True)
subj_roi_rdms_path = os.path.join(results_dir, "subj_roi_rdms")
os.makedirs(subj_roi_rdms_path, exist_ok=True)

for subj in subs:

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

        var_components = variance_partitioning(brain_rdm, all_model_rdms, all_model_names)

        # Plot Venn diagram
        k = list(var_components.keys())
        labels = [' & '.join(combo) for combo in var_components.keys()]
        subsets = [len(set().union(*(all_model_rdms[all_model_names.index(p)] for p in combo))) for combo in k]

        # helper function to round numbers in venn diagram for readability
        def decimals(n):
            return np.round(n, decimals=4)

        if len(all_model_names) == 2:
            from matplotlib_venn import venn2
            A = var_components[k[0]]  # score for 1st model
            B = var_components[k[1]]  # score for 2nd model
            A_B = var_components[k[2]]  # score of combined model
            Ab = A_B - B
            aB = A_B - A
            AB = A + B - A_B
            venn2(subsets=[Ab, aB, AB], set_labels=all_model_names, subset_label_formatter=decimals, ax=ax[(m+1)//2][(m+1)%2])
            ax[(m+1)//2][(m+1)%2].set_title(mask_name, fontsize=16, fontweight='bold')

        elif len(all_model_names) == 3:
            from matplotlib_venn import venn3
            A = var_components[k[0]]  # score for 1st model
            B = var_components[k[1]]  # score for 2nd model
            C = var_components[k[2]]  # score for 3rd model
            A_B = var_components[k[3]]
            A_C = var_components[k[4]]
            B_C = var_components[k[5]] 
            A_B_C = var_components[k[6]]
            Abc = A_B_C - B_C
            aBc = A_B_C - A_C
            ABc = A + B - A_B
            abC = A_B_C - A_B
            AbC = A + C - A_C
            aBC = B + C - B_C
            ABC = A + B + C - A_B - A_C - B_C + A_B_C
            venn3(subsets=[Abc, aBc, ABc, abC, AbC, aBC, ABC], set_labels=labels, subset_label_formatter=decimals, ax=ax[(m+1)//2][(m+1)%2])
            ax[(m+1)//2][(m+1)%2].set_title(mask_name, fontsize=16, fontweight='bold')

    plt.savefig(f'./venn_diagram_{subj}.png')
    plt.tight_layout()
    plt.close()