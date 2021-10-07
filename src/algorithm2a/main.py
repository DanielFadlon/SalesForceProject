import numpy as np
import pandas as pd
from kmeans import get_clusters
from utils import compute_distance


def order_clusters(labels, centers):

    """
    order the clusters according to distance from cluster 0

    Input:
    - labels: the labels of kmeans according to the dataset order
    - centers: the centroids of the kmeans

    Return:
    - order idx column according to data set order.
    """
    d_array = [(0, 0)]
    c_0 = centers[0]
    for idx, c in enumerate(centers[1:]):
        d_array += [(compute_distance(c_0, c), idx + 1)]
    cluster_distance = sorted(d_array, key=lambda x: x[0])

    dict_cluster_to_idx = {c[1]: idx for idx, c in enumerate(cluster_distance)}

    idx_res_column = []

    # for each row add the relevent number according to the order of the clusters
    for label in labels:
        idx_res_column += [dict_cluster_to_idx[label]]

    return idx_res_column


if __name__ == '__main__':

    """ ----------- Input --------------- 
        name of file need to the input of the algorithm, 
    """
    name_of_file = 'fourth_dataset'

    # read the data
    df = pd.read_excel(f'../Data/{name_of_file}.xlsx')

    # extract the relvent columns
    dataset = np.array(df[['Latitude', 'Longitude']])

    # get dictionary with clusters instances
    dict_of_clusters, labels, centers = get_clusters(dataset)

    # get the order of the clusters
    result_idx_column = order_clusters(labels, centers)

    # add the column to the dataFrame
    df['IDC_Index__c'] = result_idx_column

    # export to excel
    df.to_excel(f'results_{name_of_file}_(2a).xlsx', sheet_name='results')