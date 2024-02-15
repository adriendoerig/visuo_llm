import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


# Data
model_names = [
    "dnn_mpnet_rec_seedAVG_ep200_layer-1",
    "dnn_ecoset_category_layer-1",
    "thingsvision_cornet-s",
    "konkle_alexnetgn_supervised_ref12_augset1_5x",
    "brainscore_alexnet",
    "brainscore_resnet50",
    "resnext101_32x8d_wsl",
    "CLIP_resnet_images",
    "CLIP_ViT_images",
    "konkle_alexnetgn_ipcl_ref01",
    "google_simclrv1_rn50"
]

n_train_imgs = [
    48236,
    1445029,
    1281167,
    1281167,
    1281167,
    1281167,
    940000000,
    400000000,
    400000000,
    1281167,
    1281167
]

representational_agreement_ventral_ROI = [
    0.42854279,
    0.366720058,
    0.382313504,
    0.36630191,
    0.296528449,
    0.34589999,
    0.323456646,
    0.324981819,
    0.28884361,
    0.283280634,
    0.341926126
]

objective = [
    'LLM-trained',
    'category-trained',
    'category-trained',
    'category-trained',
    'category-trained',
    'category-trained',
    'weakly supervised',
    'weakly supervised',
    'weakly supervised',
    'unsupervised',
    'unsupervised'
]


# Create a color palette for each group
palette = {'LLM-trained': sns.color_palette("Oranges", objective.count('LLM-trained')),
           'category-trained': sns.color_palette("Greens", objective.count('category-trained')),
           'weakly supervised': sns.color_palette("Blues", objective.count('weakly supervised')),
           'unsupervised': sns.color_palette("RdPu", objective.count('unsupervised'))}
counts = {k: 0 for k in set(objective)}

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 10))

# Plot
for x, y, label, obj in zip(n_train_imgs, representational_agreement_ventral_ROI, model_names, objective):
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
ax.set_ylim(0.25, 0.45)

# Tight layout
plt.tight_layout()

# Save the plot
plt.savefig('trainingImgs_vs_brainMatch_scatter.svg', dpi=300)