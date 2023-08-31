import os

from .dataset_loader.make_tf_dataset import get_dataset
from .task_helper_functions import (
    get_activities_model,
    load_and_override_hparams,
    load_model_from_path,
)

os.environ[
    "CUDA_VISIBLE_DEVICES"
] = "-1"  # in case you want to run on CPU to debug or something


print("Creating model and load weights")
model_weights_path = "./save_dir"
class_names_path = "./ecoset_class_names.pickle"
n_layers = 4
dataset_path = "/mnt/klab/datasets/ecoset_square256.h5"
hparams = load_and_override_hparams(
    model_weights_path, additional_units=0, dataset=dataset_path
)  # you can override any hparam, eg. batch_size=128
net, hparams = load_model_from_path(
    model_weights_path,
    200,
    hparams=hparams,
    print_summary=True,
    test_mode=True,
)
activities_model = get_activities_model(net, n_layers, hparams)


print("Creating dataset")
dataset = get_dataset(
    hparams, dataset_path=dataset_path, dataset="test"
)  # you can set plot_generated_data=True if you want to look at the generated data

print("Evaluating on test set")
loss, acc, topk_acc = net.evaluate(dataset)
print(f"Loss: {loss}, acc: {acc}, top5acc: {topk_acc}")

for x in dataset:
    print("Getting batch...")
    batch_imgs, batch_labels, batch_class_weights = x
    print(
        batch_imgs.shape,
        batch_labels["output_time_0"].shape,
        batch_class_weights.shape,
    )  # batch_labels is a dict with one label per timestep T called 'output_time_T'

    print("Getting network outputs for batch...")
    outputs = net(batch_imgs)
    print(f"Output shape: {[o.numpy().shape for o in outputs]}")

    print("Getting layer activities...")
    layer_activities = activities_model(batch_imgs)
    print(
        "Layer activities (arranged in a list of shape [2(pre or post layer_norm)][n_layer][n_timesteps] -- but feel free to write your own activity extraction function as usual):\n"
    )
    [
        print(f"l={l}, t={t} shape: {layer[t].numpy().shape}")
        for (l, layer) in enumerate(layer_activities[0])
        for t in range(hparams["n_recurrent_steps"])
    ]

    break  # breaking out of the loop because I don't want to go through the entire test set here

print("Done.")
