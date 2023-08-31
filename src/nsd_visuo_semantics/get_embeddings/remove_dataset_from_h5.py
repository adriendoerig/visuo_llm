import h5py

orig_dataset_path = "/home/student/a/adoerig/code/datasets/ms_coco_square256_GUSE_fasttext_all_mpnet_base_v2.h5"
new_dataset_path = (
    "/home/student/a/adoerig/code/datasets/ms_coco_embeddings_square256.h5"
)
name_to_remove = "fasttext_mean_word_embeddings"

with h5py.File(orig_dataset_path, "r") as orig_dataset:
    with h5py.File(new_dataset_path, "w") as new_dataset:
        new_dataset.create_dataset(
            "categories", data=orig_dataset["categories"]
        )

        for dataset in ["train", "val", "test"]:
            group = new_dataset.create_group(dataset)
            n_samples = orig_dataset[dataset]["data"].shape[0]

            # create structures to copy over original data
            orig_subgroups = [
                s for s in orig_dataset[dataset] if s != name_to_remove
            ]
            for subgroup in orig_subgroups:
                group.create_dataset(
                    subgroup,
                    shape=orig_dataset[dataset][subgroup].shape,
                    dtype=orig_dataset[dataset][subgroup].dtype,
                    chunks=True,
                )

                new_dataset[dataset][subgroup][:] = orig_dataset[dataset][
                    subgroup
                ][:]
