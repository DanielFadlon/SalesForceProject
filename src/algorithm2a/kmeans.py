from sklearn.cluster import KMeans


def finding_the_elbow(inertia_lst):
    """
    Finding the elbow - means to find the best K for the K means according to the inertia.
    The optimal solution is available only manually.
    This function return a good K automatically.(not always the optimal)
    """
    length = len(inertia_lst) - 1

    # delta_1: index=0 -> #cluster=2 -> best_k_idx=1
    delta_1 = [inertia_lst[i] - inertia_lst[i + 1] for i in range(length)]

    # delta_2: index 0 -> #cluster=3 -> best_k_idx=2
    delta_2 = [delta_1[i] - delta_1[i + 1] for i in range(length - 1)]

    strength = [(i, delta_2[i + 1] - delta_1[i + 2]) for i in range(length - 2) if
                delta_1[i + 1] >= 0 and delta_2[i + 1] >= 0]

    best_k_idx = max(strength, key=lambda t: t[1])[0] + 2
    return best_k_idx


def choose_k(Ks, dataset):
    """
    Choosing a good k from the given list of Ks
    Return -
    - best_k_idx
    - inertia_lst - to represent that choosing make sense
    """

    # execute KMeans for each K in Ks
    inertia_lst = []
    for k in Ks:
        kmeans = KMeans(n_clusters=k, random_state=0)
        # fit the data
        _ = kmeans.fit_predict(dataset)
        # save inertia
        inertia_lst.append(kmeans.inertia_)

    best_k_idx = finding_the_elbow(inertia_lst)
    return best_k_idx, inertia_lst


def get_best_kmeans(dataset):
    """ --- this is the main of this class ---

    Input:
    - dataset: n instances with latitude and longitude

    Returns:
    - best_kmaens: kmeans object
    """

    # K to check for the KMeans
    Ks = [k for k in range(4, 150)]
    best_k_idx, inertia_lst = choose_k(Ks, dataset)
    best_k = Ks[best_k_idx]

    # get the best KMeans , n_clusters=best_k
    return best_k_idx


def seperate_instances_to_clusters(dataset, labels, best_k):
    """
    return dictionary {key = cluster_number , value = list of instances in this cluster}
    """

    # init dictionary
    dict_of_clusters = {cluster: [] for cluster in range(best_k)}

    for idx, instance in enumerate(dataset):
        dict_of_clusters[labels[idx]].append(instance)

    return dict_of_clusters


def get_clusters(dataset):
    """

    Input:
    - dataset: n instances with latitude and longitude

    Return:
    - dictionary of clusters such that:  dict[cluster] = array of cluster elements
    - labels: the result cluter of each instance in the given dataset
    """

    best_k = get_best_kmeans(dataset)
    best_kmeans = KMeans(n_clusters=best_k, random_state=0)
    # get labels
    labels = best_kmeans.fit_predict(dataset)
    # get centers  (centroids) of clusters
    centers = best_kmeans.cluster_centers_

    return seperate_instances_to_clusters(dataset, labels, best_k), labels, centers
