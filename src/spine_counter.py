"""
Authors: Kwanho Kim, Saideep Gona, Jinke Liu

Contains code for actually counting the number of spines present in a
given image given the scanned output map from the scanning step
"""

import os, argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.cluster import DBSCAN
from PIL import Image


class DBScan_Counter():

    def __init__(self, data_path, output_dir, clust_scalings, distance_metrics, epsilon_iter, min_samp_iter):
        """
        :param data_path: Path to the full scanned image data dictionary
        :param output_dir: Output directory for the scanned outputs
        :param clust_scalings: Hyperparameter iterator over scaling factor for probability element of pixel clustering vector
        :param distance_metric: Hyperparameter iterator of distance metric function objects used in DBSCAN
        :param epsilon_iter: Hyperparameter iterator over epsilon hyperparam in DBSCAN
        :param min_samp_iter: Hyperparameter iterator over minimum samples in DBSCAN
        """
        self.data = pickle.load(open(data_path, "rb"))
        # print(self.data)
        if os.path.isdir(output_dir) is False:
            os.mkdir(output_dir)
        self.output_dir = Path(output_dir)

        self.clust_scalings = clust_scalings
        self.distance_metrics = distance_metrics

        self.epsilon_iter = epsilon_iter
        self.min_samp_iter = min_samp_iter

    def convert_to_clusterables(self, scaling):
        """
        Creates clusterable objects from images and stores them in the data dict
        """

        def convert_single(image):
            imshape = image.shape
            clusterable = []

            c_ind = 0
            # plt.imshow(image)
            # plt.show()
            for x in range(imshape[0]):
                for y in range(imshape[1]):
                    # print(image[x,y], x, y)
                    if image[x, y] < 0.3:
                        # clusterable.append(1000 * np.ones((1,3)))
                        continue
                    else:
                        cur_vec = np.zeros((1, 3))
                        cur_vec[0, 0] = image[x, y] * scaling
                        cur_vec[0, 1] = x
                        cur_vec[0, 2] = y

                        clusterable.append(cur_vec[:])

                    c_ind += 1
            if len(clusterable) == 0:
                return np.array([[0, 0, 0]])
            stacked = np.stack(clusterable, axis=0)
            correct_stacked = stacked.reshape((stacked.shape[0], stacked.shape[2]))
            return correct_stacked

        for x in range(len(self.data)):
            self.data[x]["clusterable"] = convert_single(self.data[x]["scanned output"])

    def full_grid_search(self):
        """
        Runs a full grid search over the hyper-parameter space for a given accuracy metric
        """

        skip_image_factor = 1

        grid_shape = (len(self.clust_scalings), len(self.distance_metrics), len(self.epsilon_iter), len(self.min_samp_iter), len(self.data))

        full_grid_array = np.zeros((grid_shape))
        sp_count_err = np.zeros((grid_shape))

        print("Grid Shape", grid_shape)

        for w in range(grid_shape[0]):
            self.convert_to_clusterables(self.clust_scalings[w])
            for x in range(grid_shape[1]):
                for y in range(grid_shape[2]):
                    for z in range(grid_shape[3]):

                        for i in range(0,grid_shape[4], skip_image_factor):
                            print([w,x,y,z,i])
                            image_dict = self.data[i]
                            count, masses = self.count_single_image(image_dict["clusterable"],
                                                                    self.distance_metrics[x],
                                                                    self.epsilon_iter[y],
                                                                    self.min_samp_iter[z]
                                                                    )
                            err, num_spine = self.compute_accuracy(count, image_dict)
                            sp_count_err[w, x, y, z, i] = err
                            # full_grid_array[w, x, y, z, i] = err

                            clus_im = np.zeros(image_dict["image"].shape)
                            print(image_dict["clusterable"])
                            for j in range(len(image_dict["clusterable"])):
                                if masses[j] != -1:
                                    clus_im[int(image_dict["clusterable"][j][1]), int(image_dict["clusterable"][j][2])] = image_dict["scanned output"][int(image_dict["clusterable"][j][1]), int(image_dict["clusterable"][j][2])]

                            mut_info = -compute_mutual_info(clus_im, image_dict)
                            print(mut_info)
                            full_grid_array[w, x, y, z, i] = mut_info

        err_array = full_grid_array[:, :, :, :, ::skip_image_factor]
        average_accs_full = np.mean(err_array, axis=4)

        counts_array = [self.data[i]["count"] for i in range(0,grid_shape[4], skip_image_factor)]

        min_coords = np.unravel_index(np.argmin(average_accs_full), average_accs_full.shape)
        min_acc = np.min(average_accs_full)

        print(sp_count_err[min_coords].shape, sp_count_err[min_coords], np.average(sp_count_err[min_coords]))

        min_hyperparams = {
                            "cluster scaling": self.clust_scalings[min_coords[0]],
                            "distance metric": self.distance_metrics[min_coords[1]],
                            "epsilon": self.epsilon_iter[min_coords[2]],
                            "minimum samples": self.min_samp_iter[min_coords[3]]
                            }

        self.min_hyperparams = min_hyperparams

        print("Averages Shape: ", average_accs_full.shape)
        print("Minimum Val: ", min_acc)
        print("Min Hyperparams: ", min_hyperparams)

        # Plot clusters as images

        self.convert_to_clusterables(min_hyperparams["cluster scaling"])
        for i in range(0,grid_shape[4], skip_image_factor):
            image_dict = self.data[i]
            count, masses = self.count_single_image(image_dict["clusterable"],
                                                    min_hyperparams["distance metric"],
                                                    min_hyperparams["epsilon"],
                                                    min_hyperparams["minimum samples"]
                                                    )
            self.data[i]["cluster labels"] = masses

            clus_im = np.zeros(image_dict["image"].shape)
            # print(image_dict["clusterable"])
            for x in range(len(image_dict["clusterable"])):
                if self.data[i]["cluster labels"][x] != -1:
                    clus_im[int(image_dict["clusterable"][x][1]),int(image_dict["clusterable"][x][2])] = self.data[i]["cluster labels"][x]
                    self.data[i]["cluster image"] = clus_im

                    if os.path.isdir(os.path.join(self.output_dir, "prediction_figures/")) is False:
                        os.mkdir(os.path.join(self.output_dir, "prediction_figures/"))

            plt.imshow(clus_im)
            plt.savefig(os.path.join(self.output_dir, "prediction_figures/" + str(i) + "_cur_out_clust.png"))
            plt.clf()

        return sp_count_err[min_coords], counts_array

    def store_clustered_data(self):
        """
        Stores clustered image data structure as a pickle
        {
            "image",
            "centers",
            "count",
            "scanned output",
            "clusterable",
            "cluster labels",
            "cluster image"
        }
        """
        store_path = Path.joinpath(self.output_dir, "clustered_data.p")
        pickle.dump(self.data, open(store_path, "wb" ))

    def count_single_image(self, clusterable, metric, eps, min_samples):

        print("Computing Single Clustering: " + str(eps) + " , " + str(min_samples))
        # print(clusterable.shape)
        scanned_output = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(clusterable)
        clus_labels = scanned_output.labels_
        count = len(set(clus_labels)) - (1 if -1 in clus_labels else 0)

        return count, clus_labels

    def compute_accuracy(self, count, image_dict):

        raw_dist = abs(image_dict["count"] - count)

        print("count: " + str(count))
        print("true count: " + str(image_dict["count"]))
        print("error: ", str(raw_dist))
        return raw_dist, image_dict["count"]


def compute_mutual_info(pred_label, original_image_obj):
    im = original_image_obj["image"]
    centers = original_image_obj["centers"]
    true_label = expert_labeled_spines(im, centers)
    return adjusted_mutual_info_score(true_label.flatten(), pred_label.flatten())


def expert_labeled_spines(ori_arr, centers):

    result = np.zeros(shape=ori_arr.shape)

    for c in centers:
        c = (int(c[0]), int(c[1]))

        result[int(c[1])][int(c[0])] = 255
        result[int(c[1])][int(c[0]) + 1] = 255
        result[int(c[1])][int(c[0]) - 1] = 255

        result[int(c[1]) + 1][int(c[0])] = 255
        result[int(c[1]) + 1][int(c[0]) - 1] = 255
        result[int(c[1]) + 1][int(c[0]) + 1] = 255

        result[int(c[1]) - 1][int(c[0])] = 255
        result[int(c[1]) - 1][int(c[0]) + 1] = 255
        result[int(c[1]) - 1][int(c[0]) - 1] = 255

    return result


def plot_hist(arr, bins, x_label, outname):

    plt.figure()

    plt.rc('axes', linewidth=2)
    plt.rc('font', weight='bold')
    axis_label_properties = {
        'family': 'sans-serif',
        'weight': 'bold',
        'size': 24}
    plt.tick_params(axis='both', which='both', labelsize=15, width=3, pad=3, direction='in', top=False, right=False)

    h, b, patches = plt.hist(arr, bins=bins-0.5, density=True, rwidth=0.5)
    plt.ylabel("Probability", **axis_label_properties)
    plt.xlabel(x_label, **axis_label_properties)
    plt.xticks(bins)

    plt.tight_layout()
    plt.savefig(outname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processing of scanned output maps into actual dendritic spine counts")
    parser.add_argument("scans_path", help="Output path of scanned image maps")
    parser.add_argument("output_dir", help="Output directory for counting output")
    args = parser.parse_args()

    # clust_scaling_iter = [2*x for x in range(0,5)]
    # distance_metric_iter = ["euclidean", "manhattan"]
    # eps_iter = [x for x in range(1,5)]
    # min_samp_iter = [10*x for x in range(4, 8)]

    clust_scaling_iter = [8]
    distance_metric_iter = ["euclidean"]
    eps_iter = [4]
    min_samp_iter = [40]

    counter = DBScan_Counter(args.scans_path, args.output_dir, clust_scaling_iter, distance_metric_iter, eps_iter,
                             min_samp_iter)
    errors, num_spines = counter.full_grid_search()

    print(errors.shape, errors)
    print(num_spines)

    max_err = max(errors)
    max_num_sp = max(num_spines)

    plot_hist(errors, np.arange(max_err+2), "|True count - Predicted count|", os.path.join(args.output_dir, "err_hist.png"))
    # plot_hist(errors, np.arange(max_err + 2), "Mutual Information",
    #           os.path.join(args.output_dir, "err_hist.png"))
    plot_hist(num_spines, np.arange(max_num_sp+2), "Number of spines", os.path.join(args.output_dir, "num_spine_hist.png"))

    # python scanner.py 40 C:\Users\Saideep\Documents\Github_Repos\MSCB_Sem1\Deep_Learning\Project\Labeled_Spines_Tavita\ C:\Users\Saideep\Documents\Github_Repos\MSCB_Sem1\Deep_Learning\Project\counting_spines\src\training_sessions\2019-04-1515_00_29\weights.pt C:\Users\Saideep\Documents\Github_Repos\MSCB_Sem1\Deep_Learning\Project\counting_spines\src\training_sessions\2019-04-1515_00_29\
