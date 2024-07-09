

def get_modelName2Path_dict(networks_basedir):

    shared_model_prefix = "blt_vNet_half_channels"

    # specify where each set of nsd embeddings is saved
    modelname2path = {
        "multihot_rec": f"{networks_basedir}/{shared_model_prefix}_multihot_Dec23",
        "mpnet_rec": f"{networks_basedir}/{shared_model_prefix}_mpnet_Dec23",
        "mpnet_resnet50": f"{networks_basedir}/resnet50_mpnetTrained_Feb24",
        "multihot_resnet50": f"{networks_basedir}/resnet50_multihotTrained_Feb24",
        "sceneCateg_resnet50": f"{networks_basedir}/resnet50_places365Trained_Feb24",
        "dnn_ecoset_category": f"/share/klab/adoerig/adoerig/semantics_paper_nets/semantics_paper_ecoset_nets/TRAINED_ecoset_blt_vNet_categories_lr0.05_adamE0.1_reg1e-06_GN"
    }

    for seed in range(1,11):
        for modelname in ["multihot_rec", "mpnet_rec"]:
            modelname2path[f"{modelname}_seed{seed}"] = f"{modelname2path[modelname]}_seed{seed}"

    modelname2path["default"] = modelname2path["mpnet_rec"]

    return modelname2path