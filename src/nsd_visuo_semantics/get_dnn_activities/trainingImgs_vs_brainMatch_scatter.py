'''Create the scatter plots found in the paper's last figure. Here, the data is
input by hand based on results computed in the ../roi_analyses scripts'''

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

for ROI_to_plot in ['ventral', 'lateral', 'parietal']:
    # Data
    model_names = [
        "dnn_mpnet_rec_seedAVG_ep200_layer-1",  
        "dnn_ecoset_category_layer-1",  
        "thingsvision_cornet-s",  
        "nf_resnet50",  
        "konkle_alexnetgn_supervised_ref12_augset1_5x",  
        "brainscore_alexnet",  
        "brainscore_resnet50",  
        "sceneCateg_resnet50_finalLayer",  
        "taskonomy_scenecat_resnet50",  
        "resnext101_32x8d_wsl",  
        "CLIP_resnet_images",  
        "CLIP_ViT_images",  
        "konkle_alexnetgn_ipcl_ref01",  
        "google_simclrv1_rn50",  
    ]

    n_train_imgs = [
        48236,
        1445029,
        1281167,
        1281167,
        1281167,
        1281167,
        1281167,
        1803460,
        4500000,
        940000000,
        400000000,
        400000000,
        1281167,
        1281167,
        1803460,
        4500000,
    ]

    representational_agreement = {
    'ventral': [
        0.42854279,
        0.366720058,
        0.382313504,
        0.35134408538350026,
        0.36630191,
        0.296528449,
        0.34589999,
        0.33448216816515997,
        0.111603328901153,
        0.323456646,
        0.324981819,
        0.28884361,
        0.283280634,
        0.341926126,
    ],
    'parietal': [
        0.3890043514558076,
        0.3400931264826793,
        0.3792411619608211,
        0.350701305965213,
        0.35607581892552764,
        0.30501025391725256,
        0.34846631652376153,
        0.3465301251155454,
        0.13023840210512244,
        0.32464882920851196,
        0.3081016216480478,
        0.2782391070501645,
        0.3421722780958887,
        0.29461954131848833
    ],
    'lateral': [
        0.46840888014768267,
        0.3677865331591134, 
        0.360710906810104,
        0.3366029060337935,
        0.33007117464426894,
        0.26104462762159614,
        0.3292908437291413,
        0.30313205930364595,
        0.07394702561181399,
        0.31161669465559994,
        0.33325257156878696,
        0.3040260508800133,
        0.234671912101812,
        0.30233843370182767
    ]
    }

    objective = [
        'LLM-trained',
        'object categorisation',
        'object categorisation',
        'object categorisation',
        'object categorisation',
        'object categorisation',
        'object categorisation',
        'scene categorisation',
        'scene categorisation',
        'weakly supervised',
        'weakly supervised',
        'weakly supervised',
        'unsupervised',
        'unsupervised',
    ]


    # Create a color palette for each group
    palette = {'LLM-trained': 'red',
            'object categorisation': sns.color_palette("Greens", objective.count('object categorisation')),
            'scene categorisation': sns.color_palette("Oranges", objective.count('scene categorisation')),
            'weakly supervised': sns.color_palette("Blues", objective.count('weakly supervised')),
            'unsupervised': sns.color_palette("RdPu", objective.count('unsupervised'))}
    counts = {k: 0 for k in set(objective)}

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 9.5))

    # Plot
    for x, y, label, obj in zip(n_train_imgs, representational_agreement[ROI_to_plot], model_names, objective):
        color = palette[obj][counts[obj]]
        counts[obj] += 1
        ax.scatter(x, y, label=label, color=color, edgecolors='white', linewidth=1, s=600)
        
    # Add legend
    ax.legend(frameon=False)

    # Set labels and title
    ax.set_xlabel('Number of Training Images (log scale)', fontsize=12)
    ax.set_ylabel('Representational Agreement (Ventral ROI)\n[pearson correlation, normalized]', fontsize=12)

    # Set x-axis to log scale
    ax.set_xscale('log')
    # ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'1e{int(pos)}'))

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Customize tick parameters
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=8)

    # Set limits of the axes to zoom out
    ax.set_xlim(1e4, 1.3*1e9)
    ax.set_ylim(min(representational_agreement[ROI_to_plot])-0.05, max(representational_agreement[ROI_to_plot])+0.05)
    # ax.set_ylim(0.05, 0.5)

    # Tight layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'trainingImgs_vs_brainMatch_scatter_{ROI_to_plot}.svg', dpi=300)