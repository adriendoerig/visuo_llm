import torch, h5py, pickle
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import pprint

# Dataset h5 path
dataset_path_places365 = '/share/klab/datasets/places365_small/places365_small_first73000train.h5' 
dataset_path_coco_nsd = "/share/klab/datasets/ms_coco_nsd_datasets/ms_coco_embeddings_square256.h5" 
dnn_activations_path = '/share/klab/adoerig/adoerig/nsd_visuo_semantics/examples/dnn_extracted_activities'

# Define base models and transfer targets
# base_models = ['mpnet_rec_finalLayer', 'multihot_rec_finalLayer']  # I WAS LIKELY SHUFFLING THE DATASET WHEN GATHERING ACTIVITES, DO NOT USE, OR RERUN
# base_models = ['mpnet_rec_finalLayerGAP', 'multihot_rec_finalLayerGAP']  # I WAS LIKELY SHUFFLING THE DATASET WHEN GATHERING ACTIVITES, DO NOT USE, OR RERUN
# base_models = ['mpnet_rec', 'multihot_rec']  # trained on coco, activities gathered on coco
base_models = ['mpnet_rec_places365', 'multihot_rec_places365']  # trained on coco, activities gathered on places365
# transfer_targets = ['mpnet', 'multihot', 'places365']
transfer_targets = ['places365']
seeds = range(1, 9)
avg_space = False  # only applicable if using finalLayer models
shuffle_labels = False  # shuffle labels to check if the model is learning something meaningful

# hyperparams
adam_epsilon = 1e-1
lr = 5e-2
num_epochs = 200
batch_size = 96  # same as we used to train the big models

# Initialize the main dictionary
output_dict = {}

# Define train and validation performance log
train_log = {}
val_log = {}

for base_model in base_models:

    output_dict[base_model] = {}

    for target in transfer_targets:

        output_dict[base_model][target] = {}

        if target == 'places365':
            # get targets
            print(f'Loading targets for {base_model} and {target}')
            with h5py.File(dataset_path_places365, 'r') as f:
                y_train_np = f['train']['labels'][:-2000]
                y_val_np = f['train']['labels'][-2000:]
                # convert to one-hot
                y_train_np = np.eye(365)[y_train_np]
                y_val_np = np.eye(365)[y_val_np]
        else:
            if target == 'mpnet':
                dataset_target = 'all_mpnet_base_v2_mean_embeddings'
            elif target == 'multihot':
                dataset_target = 'img_multi_hot'
            else:
                raise ValueError(f'Invalid target: {target}')
            # get targets
            print(f'Loading targets for {base_model} and {target}')
            with h5py.File(dataset_path_coco_nsd, 'r') as f:
                y_train_np = f['test'][dataset_target][:-2000]
                y_val_np = f['test'][dataset_target][-2000:]

        if shuffle_labels:
            print('Shuffling labels')
            np.random.seed(0)
            y_train_np = np.random.permutation(y_train_np)
            y_val_np = np.random.permutation(y_val_np)

        for seed in seeds:

            if '_finalLayer' in base_model:
                if 'GAP' in base_model and 'places365' not in base_model:
                    base_name = base_model.split("_finalLayerGAP")[0]
                    this_model_name = f'{base_name}_seed{seed}_finalLayerGAP'
                    print(f'Loading activations for {this_model_name}')
                elif 'places365_finalLayerGAP' in base_model:
                    base_name = base_model.split("_places365_finalLayerGAP")[0]
                    this_model_name = f'{base_name}_seed{seed}_places365_finalLayerGAP'
                    print(f'Loading activations for {this_model_name}')
                else:
                    base_name = base_model.split("_finalLayer")[0]
                    this_model_name = f'{base_name}_seed{seed}_finalLayer'
                    print(f'Loading activations for {this_model_name}')
                    
                with open(f'{dnn_activations_path}/{this_model_name}_nsd_image_features_train.pkl', 'rb') as f:
                    X_train_np = pickle.load(f)
                X_val_np = X_train_np[-2000:]
                X_train_np = X_train_np[:-2000]
                if avg_space:
                    X_train_np = np.reshape(X_train_np, (X_train_np.shape[0], 16, 16, 512))
                    X_train_np = np.mean(X_train_np, axis=(1, 2))
                    X_val_np = np.reshape(X_val_np, (X_val_np.shape[0], 16, 16, 512))
                    X_val_np = np.mean(X_val_np, axis=(1, 2))
            else:
                if 'places365' in base_model:
                    base_name = base_model.split("_places365")[0]
                    this_name = f'{base_name}_seed{seed}_places365_nsd_activations_epoch200_train.h5'
                else:
                    this_name = f'{base_model}_seed{seed}_nsd_activations_epoch200.h5'
                print(f'Loading activations for {this_name}')
                with h5py.File(f'{dnn_activations_path}/{this_name}', 'r') as f:
                    X_train_np = f['layernorm_layer_9_time_5'][:-2000]
                    X_val_np = f['layernorm_layer_9_time_5'][-2000:]

            non_zero_rows_mask_train = np.any(X_train_np != 0, axis=1)
            X_train_np = X_train_np[non_zero_rows_mask_train]
            y_train_np = y_train_np[non_zero_rows_mask_train]
            non_zero_rows_mask_val = np.any(X_val_np != 0, axis=1)
            X_val_np = X_val_np[non_zero_rows_mask_val]
            y_val_np = y_val_np[non_zero_rows_mask_val]

            # Convert data to PyTorch tensors
            X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)
            X_val_tensor = torch.tensor(X_val_np, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val_np, dtype=torch.float32)   

            # Define readout model
            class MyReadout(nn.Module):
                def __init__(self, input_size, output_size, activation=None):
                    super(MyReadout, self).__init__()
                    self.linear = nn.Linear(input_size, output_size)
                    self.activation = activation

                def forward(self, x):
                    x = self.linear(x)
                    if self.activation is not None:
                        x = self.activation(x)
                    return x
                
            if target == 'mpnet':
                activation = None
            elif target == 'multihot':
                activation = nn.Sigmoid()
            elif target == 'places365':
                activation = nn.Softmax(dim=-1)
            else:
                raise ValueError(f'Invalid target: {target}')
                
            model = MyReadout(input_size=X_train_tensor.shape[1], output_size=y_train_tensor.shape[1], activation=activation)

            # Define model, loss function, and optimizer
            if target == 'places365':
                criterion = nn.CrossEntropyLoss()
            else:
                criterion = nn.CosineEmbeddingLoss()
            
            # Define Adam optimizer
            optimizer = optim.Adam(model.parameters(), lr=lr, eps=adam_epsilon)

            # make torch datasets
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            # Define the scheduler
            num_steps_per_epoch = len(train_loader)
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs*num_steps_per_epoch, eta_min=0)
        
            # Train the model
            for epoch in range(num_epochs):

                # Define progress bar
                progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)   

                for batch_X, batch_y in progress_bar:
                    model.train()
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    if target == 'places365':
                        loss = criterion(outputs, batch_y.argmax(dim=-1))
                    else:
                        loss = criterion(outputs, batch_y, torch.ones(batch_y.shape[0]))
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    # Update progress bar description
                    progress_bar.set_postfix({'Train Loss': loss.item()})

                # Log training performance
                if epoch == 0:
                    train_log[(base_model, target, seed)] = []
                train_log[(base_model, target, seed)].append(loss.item())
                
                # Evaluate on validation set
                # Evaluate on validation set
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    if target == 'places365':
                        val_loss = criterion(val_outputs, y_val_tensor.argmax(dim=-1))
                        val_acc = (val_outputs.argmax(dim=-1) == y_val_tensor.argmax(dim=-1)).float().mean().item()
                        val_top5_acc = (torch.topk(val_outputs, 5, dim=-1).indices == y_val_tensor.argmax(dim=-1).unsqueeze(1)).float().mean().item()
                        print(f'Val accuracy:{val_acc:.4f}, Val top-5 accuracy: {val_top5_acc:.4f}, Val loss: {val_loss.item():.4f}')
                    else:
                        val_loss = criterion(val_outputs, y_val_tensor, torch.ones(y_val_tensor.shape[0]))
                        print('Validation loss:', val_loss.item())
                
                # Log validation performance
                if epoch == 0:
                    val_log[(base_model, target, seed)] = []
                val_log[(base_model, target, seed)].append(val_loss.item())

            print('Saving model')
            torch.save(model.state_dict(), f'{dnn_activations_path}/transfer_{base_model}_to_{target}_seed{seed}.pth')
            # to load, build the model as usual, and then model.load_state_dict(torch.load('model.pth'))

            print('Evaluating final model')
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                if target == 'places365':
                    val_loss = criterion(val_outputs, y_val_tensor.argmax(dim=-1))
                    val_acc = (val_outputs.argmax(dim=-1) == y_val_tensor.argmax(dim=-1)).float().mean().item()
                    val_top5_acc = (torch.topk(val_outputs, 5, dim=-1).indices == y_val_tensor.argmax(dim=-1).unsqueeze(1)).float().mean().item()
                    print(f'\tFinal validation accuracy:{val_acc:.4f}, Final validation top-5 accuracy: {val_top5_acc:.4f}, Final validation loss: {val_loss.item():.4f}')
                    output_dict[base_model][target][seed] = val_acc
                
                else:
                    val_loss = criterion(val_outputs, y_val_tensor, torch.ones(y_val_tensor.shape[0]))
                    print('\tFinal validation loss:', val_loss.item())
                    cosine_sim = F.cosine_similarity(val_outputs, y_val_tensor, dim=1).mean().item()
                    print('\tFinal validation cosine similarity:', cosine_sim)
                    output_dict[base_model][target][seed] = cosine_sim
            
            print(f'Base Model: {base_model}, Target: {target}, Seed: {seed}, Epoch [{epoch+1}/{num_epochs}], '
                    f'Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
            
            # Print the training and validation logs
            # print("\nTraining Log:")
            # pprint.pprint(train_log)
            # print("\nValidation Log:")
            # pprint.pprint(val_log)

# Save the resulting dictionary
import pickle
with open(f'transfer_results{"_places365" if "places365" in transfer_targets else ""}.pkl', 'wb') as f:
    pickle.dump(output_dict, f)

# Print the resulting dictionary
print("Model Dictionary:")
pprint.pprint(output_dict)

import pdb; pdb.set_trace()