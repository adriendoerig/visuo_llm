

def get_modelName2Path_dict(networks_basedir):

    shared_model_prefix = "blt_vNet_half_channels"

    # specify where each set of nsd embeddings is saved
    modelname2path = {
        "multihot_rec_small": f"{networks_basedir}/{shared_model_prefix}_multihot_Nov23",
        "mpnet_rec_small": f"{networks_basedir}/{shared_model_prefix}_mpnet_Nov23",
        "multihot_rec": f"{networks_basedir}/{shared_model_prefix}_multihot_Dec23",
        "mpnet_rec": f"{networks_basedir}/{shared_model_prefix}_mpnet_Dec23",
        "simclr_rec": f"{networks_basedir}/simclr_{shared_model_prefix}_simclr_Nov23",
    }

    for seed in range(1,11):
        for modelname in ["multihot_rec", "mpnet_rec", 'simclr_rec']:
            modelname2path[f"{modelname}_seed{seed}"] = f"{modelname2path[modelname]}_seed{seed}"

    modelname2path["default"] = modelname2path["mpnet_rec"]

    return modelname2path