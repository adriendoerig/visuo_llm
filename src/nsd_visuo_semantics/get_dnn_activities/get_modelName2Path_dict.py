

def get_modelName2Path_dict(networks_basedir):

    shared_model_prefix = "blt_vNet_half_channels"

    # specify where each set of nsd embeddings is saved
    modelname2path = {
        "multihot_rec_small": f"{networks_basedir}/{shared_model_prefix}_multihot_Nov23",
        "mpnet_rec_small": f"{networks_basedir}/{shared_model_prefix}_mpnet_Nov23",
        "multihot_rec": f"{networks_basedir}/{shared_model_prefix}_multihot_Dec23",
        "mpnet_rec": f"{networks_basedir}/{shared_model_prefix}_mpnet_Dec23",
        "simclr_rec": f"{networks_basedir}/simclr_{shared_model_prefix}_simclr_Nov23",
        "mpnet_resnet50": f"{networks_basedir}/resnet50_mpnetTrained_Feb24",
        "multihot_resnet50": f"{networks_basedir}/resnet50_multihotTrained_Feb24",
        "sceneCateg_resnet50": f"{networks_basedir}/resnet50_places365Trained_Feb24",
    }

    for seed in range(1,11):
        for modelname in ["multihot_rec", "mpnet_rec", 'simclr_rec']:
            modelname2path[f"{modelname}_seed{seed}"] = f"{modelname2path[modelname]}_seed{seed}"

    # custom controls, etc
    modelname2path[f"multihot_rec_seed1_softmax"] = f"{networks_basedir}/{shared_model_prefix}_multihot_Dec23_seed1_softmax"
    modelname2path[f"multihot_rec_seed1_noHalfPrecBs96normAxes-1"] = f"{networks_basedir}/{shared_model_prefix}_mpnet_Dec23_seed1_noHalfPrecBs96normAxes-1"

    modelname2path["default"] = modelname2path["mpnet_rec"]

    return modelname2path