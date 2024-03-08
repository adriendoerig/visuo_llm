import os
import imageio

base_dir = '/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/decoding_analyses/all-mpnet-base-v2_results_ROIfullbrain_encodingModel/interpolate_project_embeddings'
brain_maps_png_dir = f'{base_dir}/brain_maps'
save_dir = f'{base_dir}/gifs'
os.makedirs(save_dir, exist_ok=True)

n_steps = 100
name = 'single_word_people_to_single_word_food'
subj = 'Avg'
view = 13

png_files = [f"{brain_maps_png_dir}/{name}_interp{i}_subj{subj}_view{view}.png" for i in range(n_steps)]

# Create GIF from PNG images
images = []
for png_file in png_files:
    images.append(imageio.imread(png_file))

# Save GIF
imageio.mimsave(f'{save_dir}/{name}.gif', images)
