# This function is used to create the clusters of the semantic space. It uses the KMeans algorithm to create the clusters.
# This can be done either for a presedefined number of clusters or for a range of clusters. The function returns the cluster
# In case we enter a range of clusters, the function returns the silhouette score for each sample in the range of clusters.
# We then compute the optimal number of clusters based on the silhouette score (e.g. median of the silhouette scores for each cluster)
# We will heaviliy draw on parallel processing to speed up the computation of the clusters.
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

    

def cluster_semantic_space_crossval(X, min_clusters=5, max_clusters=100, stepsize=5, seed=42, minibatchKmeans=False, rand_test_size=0.1, n_cvals=50, log_steps=False):
    """
    Cluster the semantic space using KMeans algorithm with cross-validation. 
    This is done by splitting the data into a training and test set and then computing the silhouette score on the test set.


    Parameters
    ----------
    X : np.array
        The data to cluster
    min_clusters : int
        The minimum number of clusters to test
    max_clusters : int
        The maximum number of clusters to test
    stepsize : int
        The stepsize to use when testing a range of clusters
    seed : int
        The random seed to use
    minibatchKmeans : bool
        Whether to use the minibatchKmeans algorithm
    rand_test_size : float
        The size of the test set
    n_cvals : int
        The number of cross-validation iterations
    log_steps : bool
        Whether to use a logarithmic range of clusters

    Returns
    -------
    mean_silhouette_over_k_test : np.array
        The mean silhouette score for each number of clusters and cross-validation iteration
    silhouette_samples_over_k_test : np.array
        The silhouette score for each sample for each number of clusters and cross-validation iteration
    cluster_labels_over_k : np.array
        The cluster labels for each sample for each number of clusters and cross-validation iteration
    saved_operators : dict
        The KMeans operators for each number of clusters
    solution_all_k : np.array
        The cluster labels for each sample (not just test or train set)

    """
    num_samples = X.shape[0]


    n_clusters_steps = list(np.arange(start = min_clusters, stop = max_clusters, step = stepsize))
    if log_steps:
        n_clusters_steps = [int(x) for x in np.logspace(np.log10(min_clusters), np.log10(max_clusters), num=10)]
    #n_clusters_steps.append(200)
    mean_silhouette_over_k_test   = np.full((n_cvals, len(n_clusters_steps)), np.NaN)
   

    saved_operators = dict()
    print(n_clusters_steps)
    for cluster_counter, num_clusters in enumerate(n_clusters_steps):
        for cval in range(n_cvals):
            print(num_clusters, cval)
            if minibatchKmeans:
                KMCLUSTER = MiniBatchKMeans(n_clusters=num_clusters, init='k-means++',
                                            max_iter=100, batch_size=2000, verbose=0,
                                            compute_labels=False, random_state=seed+num_clusters+cval, 
                                            tol=0.0, max_no_improvement=10, init_size=None, 
                                            n_init=1, reassignment_ratio=0.01)
            else:
                KMCLUSTER = KMeans(n_clusters=num_clusters,init='k-means++', n_init=1, max_iter=300, 
                            tol=0.0001, verbose=0,
                            random_state=seed, copy_x=True)
    
                # whether to fit and test the data on random splits of the whole mscoco dataset or wether to train on the whole
                # dataset and test on the ms-coco validation set
            X_train, X_test =train_test_split(X, test_size=rand_test_size, random_state=seed+cval)
            if cluster_counter == 0 and cval == 0:
                # adjust size of the silhouette_samples_over_k_test array
                train_size_samples = X_test.shape[0]
                silhouette_samples_over_k_test = np.full((n_cvals, len(n_clusters_steps), train_size_samples), np.NaN)
                cluster_labels_over_k = np.full((n_cvals, len(n_clusters_steps), train_size_samples), np.NaN)
          
            KMCLUSTER.fit(X_train)
            saved_operators[num_clusters] = KMCLUSTER
            cluster_labels_test = KMCLUSTER.predict(X_test)
            print('computing silhoutte') 
            cluster_labels_over_k[cval, cluster_counter, :] = cluster_labels_test
            silhouette_samples_over_k_test[cval, cluster_counter, :] =  silhouette_samples(X_test, cluster_labels_test)
            mean_silhouette_over_k_test[cval, cluster_counter] = silhouette_score(X_test, cluster_labels_test)

            # store the cluster labels for the whole dataset
            if cluster_counter == 0 and cval == 0:
                solution_all_k = dict()
            solution_all_k[num_clusters] = KMCLUSTER.predict(X)
            print('done')


    return mean_silhouette_over_k_test, silhouette_samples_over_k_test, cluster_labels_over_k, saved_operators, solution_all_k
    
        

def plot_silhouette_scores_per_k(df, out_dir, range_clusters=None, analysis_name=""):
    """
    Plot the distribution of min silhoutte per crossval iteration for each number of clusters
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    #sns.violinplot(x="k", y="silhouette_score", data=df)
    
    if range_clusters is None:
        range_clusters = df["k"].unique()
    # compute the min silhouette score per cv iteration for each number of clusters
    print(df.head())
    min_silhouette_per_k = df.groupby(["k", "cval_iteration"])["silhouette_scores"].min().reset_index()
    print(min_silhouette_per_k)
    # plot the distribution of min silhouette scores per number of clusters

    sns.violinplot(x="k", y="silhouette_scores", data=min_silhouette_per_k, label = "min over cval iterations")

    # plot the median silhouette score per number of clusters
    median_silhouette_per_k = df.groupby(["k", "cval_iteration"])["silhouette_scores"].mean().reset_index()
    sns.violinplot(x="k", y="silhouette_scores", data=median_silhouette_per_k, label = "mean over cval iterations")
    plt.legend()
    # dont repeat the enrties in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    title = f"Silhouette scores per number of clusters"
    if analysis_name:
        title = f"{analysis_name} - {title}"
    plt.title(title)
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette score")

    # save the plot
    if analysis_name == "":
        fname = f"silhouette_scores_per_k_{range_clusters}.png"
    else:
        fname = f"silhouette_scores_per_k_{analysis_name}_{range_clusters}.png"

    plt.savefig(os.path.join(out_dir, fname))
    plt.close()
    return None



if __name__ == "__main__":

    zscore = True
    zscore_suffix = "_zscored" if zscore else ""

    roi = 'visROIs' #  "sigOnly"#"visROIs" #fullBrain
    min_clusters = 4
    max_clusters = 12
    stepsize = 1
    log_steps = False
    n_cvals = 10
    seed = 42
    minibatchKmeans = False
    rand_test_size = 0.1

    data_path = "/share/klab/datasets/_for_philip_clustering_from_adrien"
    data_fname = f"encodingModelCoeffs_{roi}_filtered_data_in_subjavg.npy"
    X = np.load(os.path.join(data_path, data_fname))

    if zscore:
        X = (X - X.mean(axis=0)) / X.std(axis=0)

    out_dir = '/share/klab/adoerig/adoerig/nsd_visuo_semantics/results_dir/voxel_wise_analyses/cache/semantic_clusters'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # store the cluster labels and silhouette scores in a longformat df and save it to the out_dir
    df = pd.DataFrame(columns=["k", "cval_iteration", "cluster_labels", "silhouette_scores"])
    
    # compute the silhouette scores for a range of clusters
    silhouette_score, silhouette_samples, cluster_labels, saved_operators, lablels_all_per_k = cluster_semantic_space_crossval(X, min_clusters, max_clusters, stepsize, seed, minibatchKmeans, rand_test_size, n_cvals, log_steps=log_steps)
    # add the silhouette scores per sample to the df
    dfs_to_concat = []
    for cval in range(silhouette_samples.shape[0]):
        for k in range(silhouette_samples.shape[1]):
            k_name = k*stepsize + min_clusters
            for sample in range(silhouette_samples.shape[2]):
                dfs_to_concat.append(pd.DataFrame({"k": k_name, "cval_iteration": cval, "cluster_labels": cluster_labels[cval, k, sample],
                                                    "silhouette_scores": silhouette_samples[cval, k, sample]}, index=[0]))
    df = pd.concat(dfs_to_concat, ignore_index=True)

    # store the cluster labels for the whole dataset
    df_labels_all = pd.DataFrame(lablels_all_per_k)
    
    plot_silhouette_scores_per_k(df, out_dir, analysis_name=roi)
    print(silhouette_score)
    # store the csv 
    df.to_csv(os.path.join(out_dir, f"silhouette_scores_per_k_{roi}_{min_clusters}_{max_clusters}_{stepsize}{zscore_suffix}.csv"))
    df_labels_all.to_csv(os.path.join(out_dir, f"cluster_labels_all_{roi}_{min_clusters}_{max_clusters}_{stepsize}{zscore_suffix}.csv"))

  
  