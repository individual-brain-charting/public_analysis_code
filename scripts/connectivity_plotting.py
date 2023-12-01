import os
from glob import glob
import pandas as pd
import seaborn as sns
from ibc_public import utils_connectivity as fc
from nilearn.connectome import sym_matrix_to_vec
import numpy as np
from nilearn.connectome import vec_to_sym_matrix
from nilearn.plotting import plot_connectome, view_connectome
import matplotlib.pyplot as plt
from nilearn import datasets
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import LinearSVC

### create sc_data dataframe for native space
cache = DATA_ROOT = "/storage/store2/work/haggarwa/"
n_parcels = 200
sub_ses, _ = fc.get_ses_modality("DWI")
sc_data_native = []
for sub, session in sub_ses.items():
    data = {"subject": sub, "measure": "SC", "task": "SC"}
    path = os.path.join(DATA_ROOT, sub, session, "dwi")
    csv = glob(
        os.path.join(
            path,
            f"*connectome_schaefer{n_parcels}_individual_siftweighted.csv",
        )
    )
    matrix = pd.read_csv(csv[0], header=None).to_numpy()
    print(matrix.shape)
    matrix = sym_matrix_to_vec(matrix, discard_diagonal=True)
    data["connectivity"] = matrix
    sc_data_native.append(data)

sc_data_native = pd.DataFrame(sc_data_native)
sc_data_native.to_pickle(
    os.path.join(DATA_ROOT, f"sc_data_native_{n_parcels}")
)

### create sc_data dataframe for MNI space
sub_ses, _ = fc.get_ses_modality("DWI")
sc_data_mni = []
for sub, session in sub_ses.items():
    data = {"subject": sub, "measure": "SC", "task": "SC"}
    path = os.path.join(DATA_ROOT, sub, session, "dwi")
    csv = glob(
        os.path.join(path, "*connectome_schaefer400_MNI152_siftweighted.csv")
    )
    matrix = pd.read_csv(csv[0], header=None).to_numpy()
    print(matrix.shape)
    matrix = sym_matrix_to_vec(matrix, discard_diagonal=True)
    data["connectivity"] = matrix
    sc_data_mni.append(data)

sc_data_mni = pd.DataFrame(sc_data_mni)
sc_data_mni.to_pickle(os.path.join(DATA_ROOT, "sc_data_mni_new"))


def get_lower_tri_heatmap(
    df,
    figsize=(11, 9),
    cmap="viridis",
    annot=False,
    title=None,
    ticks=None,
    labels=None,
    grid=False,
    output="matrix.png",
    triu=False,
    diag=False,
    tril=False,
):
    mask = np.zeros_like(df)
    mask[np.triu_indices_from(mask)] = triu

    mask[np.tril_indices_from(mask)] = tril

    # Want diagonal elements as well
    mask[np.diag_indices_from(mask)] = diag

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=figsize)

    # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns_plot = sns.heatmap(
        df,
        mask=mask,
        cmap=cmap,
        # cmap=cmap,
        # vmax=0.008,
        # vmin=0.003,
        # center=0,
        square=True,
        linewidths=0,
        cbar_kws={"shrink": 0.5},
        ax=ax,
        annot=annot,
        fmt="",
        # annot_kws={
        #     "backgroundcolor": "white",
        #     "color": "black",
        #     "bbox": {
        #         "alpha": 0.5,
        #         "color": "white",
        #     },
        # },
    )
    if grid:
        ax.grid(grid, color="black", linewidth=0.5)
    else:
        ax.grid(grid)
    if labels is not None and ticks is None:
        ax.set_xticklabels(labels, rotation=40, ha="right")
        ax.set_yticklabels(labels, rotation=0)
    elif labels is not None and ticks is not None:
        ax.set_xticks(ticks, labels)
        ax.set_yticks(ticks, labels)
        ax.tick_params(left=True, bottom=True)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    if title is not None:
        ax.set_title(title)

    # save to file
    fig = sns_plot.get_figure()
    fig.savefig(f"{output}.png", bbox_inches="tight")
    fig.savefig(f"{output}.svg", bbox_inches="tight", transparent=True)
    plt.close("all")


### fit classifiers to get weights ###
DATA_ROOT = cache = "/storage/store2/work/haggarwa/"
n_parcels = 200
if n_parcels == 400:
    func_data_path = os.path.join(cache, "connectomes2")
    output_dir = os.path.join(DATA_ROOT, "weights")
elif n_parcels == 200:
    func_data_path = os.path.join(cache, "connectomes_200_parcels")
    output_dir = os.path.join(DATA_ROOT, "weights_200")
os.makedirs(output_dir, exist_ok=True)
func_data = pd.read_pickle(func_data_path)
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
measures = ["correlation", "partial correlation"]
classify = ["Tasks", "Subjects", "Runs"]
for clas in tqdm(classify, desc="Classify", position=0):
    for cov in tqdm(cov_estimators, desc="Estimator", position=1):
        for measure in measures:
            weights_file = os.path.join(
                output_dir, f"{clas}_{cov}_{measure}_weights.npy"
            )
            if os.path.exists(weights_file):
                print(f"skipping {cov} {measure}, already done")
                continue
            else:
                if clas == "Tasks":
                    classes = func_data["tasks"].to_numpy(dtype=object)
                elif clas == "Subjects":
                    classes = func_data["subject_ids"].to_numpy(dtype=object)
                elif classify == "Runs":
                    data["run_task"] = data["run_labels"] + "_" + data["tasks"]
                    classes = data["run_task"].to_numpy(dtype=object)
                data = np.array(func_data[f"{cov} {measure}"].values.tolist())
                classifier = LinearSVC(max_iter=100000, dual="auto").fit(
                    data, classes
                )
                np.save(weights_file, classifier.coef_)


### network pair SVC weight matrices ###
# load the data
DATA_ROOT = cache = "/storage/store2/work/haggarwa/"

n_parcels = 200
if n_parcels == 400:
    weight_dir = os.path.join(DATA_ROOT, "weights")
    output_dir = os.path.join(DATA_ROOT, "weight_plots")
elif n_parcels == 200:
    weight_dir = os.path.join(DATA_ROOT, "weights_200")
    output_dir = os.path.join(DATA_ROOT, "weight_plots_200")
os.makedirs(output_dir, exist_ok=True)
# get atlas for yeo network labels
atlas = datasets.fetch_atlas_schaefer_2018(
    data_dir=cache, resolution_mm=2, n_rois=n_parcels
)
networks = atlas["labels"].astype("U")
hemi_network_labels = []
network_labels = []
for network in networks:
    components = network.split("_")
    hemi_network = "_".join(components[1:3])
    hemi_network_labels.append(hemi_network)
    network_label = components[2]
    if network_label == "Vis":
        network_label = "Visual"
    elif network_label == "Cont":
        network_label = "FrontPar"
    elif network_label == "SalVentAttn":
        network_label = "VentAttn"
    network_labels.append(network_label)
le = preprocessing.LabelEncoder()
labels = le.fit_transform(network_labels)
# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
# what to classify
classify = ["Tasks", "Subjects", "Runs"]
for clas in classify:
    for cov in cov_estimators:
        for measure in measures:
            cov_measure = f"{cov}_{measure}"
            # load and average weights
            try:
                weights = np.load(
                    os.path.join(
                        weight_dir, f"{clas}_{cov_measure}_weights.npy"
                    )
                )
            except FileNotFoundError:
                print(f"skipping {clas} {cov_measure}")
                continue
            # weights_ = []
            # for i in list(weights):
            #     print(i.shape)
            #     # if i.shape[0] == 11:
            #     #     continue
            #     # else:
            #     #     weights_.append(i)
            #     weights_.append(i)
            # weights = np.dstack(weights_)
            # weights = np.mean(weights, axis=2)
            norm_weights = np.sqrt(np.sum(weights**2, axis=0))
            norm_weight_mat = vec_to_sym_matrix(
                norm_weights, diagonal=np.zeros(n_parcels)
            )

            network_pair_weights = np.zeros((7, 7))
            unique_labels = np.unique(labels)
            # get network pair weights
            for network_i in unique_labels:
                index_i = np.where(labels == network_i)[0]
                for network_j in unique_labels:
                    index_j = np.where(labels == network_j)[0]
                    if network_i != network_j:
                        network_pair_weight = np.mean(
                            norm_weight_mat[index_i, :][:, index_j].flatten()
                        )
                    elif network_i == network_j:
                        # continue
                        weight_indices = np.tril_indices_from(
                            norm_weight_mat[index_i, :][:, index_j], k=-1
                        )
                        network_pair_weight = norm_weight_mat[weight_indices]
                        network_pair_weight = np.mean(network_pair_weight)
                    network_pair_weights[network_i][
                        network_j
                    ] = network_pair_weight
            sns.set_context("notebook")
            get_lower_tri_heatmap(
                network_pair_weights,
                figsize=(5, 5),
                cmap="hot_r",
                labels=le.inverse_transform(unique_labels),
                output=os.path.join(
                    output_dir, f"{clas}_{cov_measure}_weights"
                ),
                triu=True,
                title=f"{cov} {measure}",
            )


### fc-fc similarity, sub-spec matrices ###
# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
# what to classify
classify = ["Tasks", "Subjects", "Runs"]
# tasks
tasks = ["RestingState", "Raiders", "GoodBadUgly", "MonkeyKingdom", "Mario"]
# load the data
DATA_ROOT = "/storage/store2/work/haggarwa/"
n_parcels = 400
if n_parcels == 400:
    results_dir = "fc_similarity_20231106-125501"
elif n_parcels == 200:
    results_dir = "fc_similarity_20231117-164946"
similarity_data = pd.read_pickle(
    os.path.join(DATA_ROOT, results_dir, "results.pkl")
)
# create matrices showing similarity between tasks
if n_parcels == 400:
    output_dir = os.path.join(DATA_ROOT, "similarity_plots")
elif n_parcels == 200:
    output_dir = os.path.join(DATA_ROOT, "similarity_plots_200")
os.makedirs(output_dir, exist_ok=True)
for cov in cov_estimators:
    for measure in measures:
        similarity_values = np.zeros((len(tasks), len(tasks)))
        similarity_tasks = np.zeros((len(tasks), len(tasks)), dtype=object)
        for i, task1 in enumerate(tasks):
            for j, task2 in enumerate(tasks):
                if i == j:
                    similarity = 1
                elif i > j:
                    similarity = similarity_values[j][i]
                else:
                    mask = (
                        (similarity_data["task1"] == task1)
                        & (similarity_data["task2"] == task2)
                        & (similarity_data["measure"] == f"{cov} {measure}")
                        & (similarity_data["centering"] == "uncentered")
                    )
                    matrix = similarity_data[mask]["matrix"].to_numpy()[0]
                    similarity = np.mean(matrix)
                similarity_values[i][j] = similarity
                similarity_tasks[i][j] = (task1, task2)
        sns.set_context("notebook")
        get_lower_tri_heatmap(
            similarity_values,
            # cmap="Reds",
            figsize=(5, 5),
            labels=tasks,
            output=os.path.join(output_dir, f"similarity_{cov}_{measure}"),
            triu=True,
            diag=True,
            tril=False,
            title=f"{cov} {measure}",
        )
# create matrices showing subject-specificity of tasks
if n_parcels == 400:
    output_dir = os.path.join(DATA_ROOT, "subspec_plots")
elif n_parcels == 200:
    output_dir = os.path.join(DATA_ROOT, "subspec_plots_200")
os.makedirs(output_dir, exist_ok=True)
for cov in cov_estimators:
    for measure in measures:
        similarity_values = np.zeros((len(tasks), len(tasks)))
        similarity_type = np.zeros((len(tasks), len(tasks)), dtype=object)
        for centering in ["uncentered", "centered"]:
            for i, task1 in enumerate(tasks):
                for j, task2 in enumerate(tasks):
                    if i < j:
                        mask = (
                            (similarity_data["task1"] == task1)
                            & (similarity_data["task2"] == task2)
                            & (
                                similarity_data["measure"]
                                == f"{cov} {measure}"
                            )
                            & (similarity_data["centering"] == centering)
                        )
                        matrix = similarity_data[mask]["matrix"].to_numpy()[0]
                        n_subs = len(
                            similarity_data[mask]["kept_subjects"].to_numpy()[
                                0
                            ]
                        )
                        matrix = matrix.reshape((n_subs, n_subs))
                        same_sub = np.mean(np.diagonal(matrix, offset=0))
                        upper_tri = np.triu(matrix, k=1).flatten()
                        lower_tri = np.tril(matrix, k=-1).flatten()
                        cross_sub = np.concatenate((upper_tri, lower_tri))
                        cross_sub = np.mean(cross_sub)
                        similarity_values[i][j] = same_sub
                        similarity_values[j][i] = cross_sub
                        similarity_type[i][j] = ""
                        similarity_type[j][i] = ""
                    elif i == j:
                        similarity_values[i][j] = np.nan
                        similarity_type[i][j] = ""
                    else:
                        continue
                    similarity_type[1][3] = "Within-subject\nsimilarity"
                    similarity_type[3][1] = "Across-subject\nsimilarity"
            similarity_type = pd.DataFrame(similarity_type)
            similarity_type = similarity_type.astype("str")
            similarity_annot = similarity_values.round(2)
            similarity_annot = similarity_annot.astype("str")
            similarity_annot[1][3] = "Within\nsubs"
            similarity_annot[3][1] = "Across\nsubs"
            sns.set_context("notebook")
            get_lower_tri_heatmap(
                similarity_values,
                figsize=(5, 5),
                # cmap="RdBu_r",
                annot=similarity_annot,
                labels=tasks,
                output=os.path.join(
                    output_dir, f"subspec_{cov}_{measure}_{centering}"
                ),
                title=f"{cov} {measure}",
            )

### connectivity plots ###
# get atlas for yeo network labels
cache = DATA_ROOT = "/storage/store2/work/haggarwa/"
atlas = datasets.fetch_atlas_schaefer_2018(
    data_dir=cache, resolution_mm=2, n_rois=400
)
networks = atlas["labels"].astype("U")
hemi_network_labels = []
for network in networks:
    components = network.split("_")
    hemi_network = "_".join(components[1:3])
    hemi_network_labels.append(hemi_network)
ticks = []
for i, label in enumerate(network_labels):
    if label != network_labels[i - 1]:
        ticks.append(i)
# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
# what to classify
classify = ["Tasks", "Subjects", "Runs"]
# tasks
tasks = ["RestingState", "Raiders", "GoodBadUgly", "MonkeyKingdom", "Mario"]
# load the data
fc_data = pd.read_pickle(os.path.join(DATA_ROOT, "connectomes2"))
_, uniq_idx = np.unique(hemi_network_labels, return_index=True)
hemi_network_labels = np.array(hemi_network_labels)[np.sort(uniq_idx)]
sns.set_context("notebook", font_scale=1.05)
mats_dir = os.path.join(DATA_ROOT, "connectivity_matrices")
brain_dir = os.path.join(DATA_ROOT, "brain_connectivity")
html_dir = os.path.join(DATA_ROOT, "brain_connectivity_html")
os.makedirs(mats_dir, exist_ok=True)
os.makedirs(brain_dir, exist_ok=True)
os.makedirs(html_dir, exist_ok=True)
coords = pd.read_csv(
    os.path.join(
        cache,
        "schaefer_2018",
        "Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv",
    )
)[["R", "A", "S"]].to_numpy()
for sub in tqdm(np.unique(fc_data["subject_ids"])):
    for task in tasks:
        for cov in cov_estimators:
            for measure in measures:
                try:
                    matrix = vec_to_sym_matrix(
                        np.mean(
                            np.vstack(
                                list(
                                    fc_data[
                                        (fc_data["subject_ids"] == sub)
                                        & (fc_data["tasks"] == task)
                                    ][f"{cov} {measure}"]
                                )
                            ),
                            axis=0,
                        ),
                        diagonal=np.ones(400),
                    )
                except ValueError:
                    print(f"{sub} {task} {cov} {measure} does not exist")
                    continue
                get_lower_tri_heatmap(
                    matrix,
                    title=f"{sub} {task}",
                    output=os.path.join(
                        mats_dir, f"{sub}_{task}_{cov}_{measure}"
                    ),
                    ticks=ticks,
                    labels=hemi_network_labels,
                    grid=True,
                    diag=True,
                    triu=True,
                )
                # f = plt.figure(figsize=(9, 4))
                # plot_connectome(
                #     matrix,
                #     coords,
                #     edge_threshold="99.8%",
                #     title=f"{sub} {task}",
                #     node_size=25,
                #     figure=f,
                #     colorbar=True,
                #     output_file=os.path.join(
                #         brain_dir,
                #         f"{sub}_{task}_{cov}_{measure}_connectome.png",
                #     ),
                # )
                # threshold = np.percentile(matrix, 99.8)
                # matrix_thresholded = np.where(matrix > threshold, matrix, 0)
                # max_ = np.max(matrix)
                # three_d = view_connectome(
                #     matrix,
                #     coords,
                #     # edge_threshold="99.8%",
                #     symmetric_cmap=False,
                #     title=f"{sub} {task}",
                # )
                # three_d.save_as_html(
                #     os.path.join(
                #         html_dir,
                #         f"{sub}_{task}_{cov}_{measure}_connectome.html",
                #     )
                # )
                plt.close("all")


### structural connectivity plots ###
# get atlas for yeo network labels
cache = DATA_ROOT = "/storage/store2/work/haggarwa/"
atlas = datasets.fetch_atlas_schaefer_2018(
    data_dir=cache, resolution_mm=2, n_rois=400
)
networks = atlas["labels"].astype("U")
hemi_network_labels = []
for network in networks:
    components = network.split("_")
    hemi_network = "_".join(components[1:3])
    hemi_network_labels.append(hemi_network)
ticks = []
for i, label in enumerate(hemi_network_labels):
    if label != hemi_network_labels[i - 1]:
        ticks.append(i)
# load the data
sc_data = pd.read_pickle(os.path.join(DATA_ROOT, "sc_data_native_new"))
_, uniq_idx = np.unique(hemi_network_labels, return_index=True)
hemi_network_labels = np.array(hemi_network_labels)[np.sort(uniq_idx)]
sns.set_context("notebook", font_scale=1.05)
mats_dir = os.path.join(DATA_ROOT, "sc_matrices")
brain_dir = os.path.join(DATA_ROOT, "sc_glass_brain")
html_dir = os.path.join(DATA_ROOT, "sc_html")
os.makedirs(mats_dir, exist_ok=True)
os.makedirs(brain_dir, exist_ok=True)
os.makedirs(html_dir, exist_ok=True)
coords = pd.read_csv(
    os.path.join(
        cache,
        "schaefer_2018",
        "Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv",
    )
)[["R", "A", "S"]].to_numpy()
for sub in tqdm(np.unique(sc_data["subject"])):
    try:
        matrix = vec_to_sym_matrix(
            np.mean(
                np.vstack(
                    list(sc_data[sc_data["subject"] == sub]["connectivity"])
                ),
                axis=0,
            ),
            diagonal=np.ones(400),
        )
    except ValueError:
        print(f"{sub} does not exist")
        continue
    # get_lower_tri_heatmap(
    #     matrix,
    #     title=f"{sub}",
    #     output=os.path.join(mats_dir, f"{sub}"),
    #     ticks=ticks,
    #     labels=hemi_network_labels,
    #     grid=True,
    #     diag=True,
    #     triu=True,
    # )
    f = plt.figure(figsize=(9, 4))
    plot_connectome(
        matrix,
        coords,
        edge_threshold="99.8%",
        title=f"{sub}",
        node_size=25,
        figure=f,
        colorbar=True,
        output_file=os.path.join(
            brain_dir,
            f"{sub}_connectome.png",
        ),
    )
    threshold = np.percentile(matrix, 99.8)
    matrix_thresholded = np.where(matrix > threshold, matrix, 0)
    max_ = np.max(matrix)
    three_d = view_connectome(
        matrix,
        coords,
        edge_threshold="99.8%",
        symmetric_cmap=False,
        title=f"{sub}",
    )
    three_d.save_as_html(
        os.path.join(
            html_dir,
            f"{sub}_connectome.html",
        )
    )
    plt.close("all")


### pooled or multi task classification accuracies ###
cache = DATA_ROOT = "/storage/store/work/haggarwa/"
DATA_ROOT2 = "/storage/store2/work/haggarwa/"

n_parcels = 200
if n_parcels == 400:
    results_dir = "fc_classification_20231013-135814"  # 400 parcels results
    results_pkl = os.path.join(
        DATA_ROOT, results_dir, "all_results.pkl"
    )  # 400 parcels results
    output_dir = "classification_plots"  # 400 parcels results
elif n_parcels == 200:
    results_dir = "fc_classification_20231115-140922"  # 200 parcels results
    results_pkl = os.path.join(
        DATA_ROOT2, results_dir, "all_results.pkl"
    )  # 200 parcels results
    output_dir = "classification_plots_200"  # 200 parcels results
output_dir = os.path.join(DATA_ROOT2, output_dir)
os.makedirs(output_dir, exist_ok=True)
df = pd.read_pickle(results_pkl)

# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
# what to classify
classify = ["Tasks", "Subjects", "Runs"]


for clas in classify:
    df_ = df[df["classes"] == clas]
    df_.reset_index(inplace=True, drop=True)
    if n_parcels == 400:
        df_.drop(columns=["weights"], inplace=True)

    balanced_accuracies = []
    dummy_balanced_accuracies = []
    for _, row in df_.iterrows():
        balanced_accuracy = balanced_accuracy_score(
            row["true_class"], row["LinearSVC_predicted_class"]
        )
        dummy_balanced_accuracy = balanced_accuracy_score(
            row["true_class"], row["Dummy_predicted_class"]
        )
        balanced_accuracies.append(balanced_accuracy)
        dummy_balanced_accuracies.append(dummy_balanced_accuracy)
    df_["balanced_accuracy"] = balanced_accuracies
    df_["dummy_balanced_accuracy"] = dummy_balanced_accuracies

    for how_many in ["all", "three"]:
        if how_many == "all":
            order = [
                "Unregularized correlation",
                "Unregularized partial correlation",
                "Ledoit-Wolf correlation",
                "Ledoit-Wolf partial correlation",
                "Graphical-Lasso correlation",
                "Graphical-Lasso partial correlation",
            ]
        elif how_many == "three":
            order = [
                "Unregularized correlation",
                "Ledoit-Wolf correlation",
                "Graphical-Lasso partial correlation",
            ]
        ax_score = sns.barplot(
            y="connectivity",
            x="balanced_accuracy",
            data=df_,
            orient="h",
            palette=sns.color_palette()[0:1],
            order=order,
            facecolor=(0.4, 0.4, 0.4, 1),
        )
        for i in ax_score.containers:
            plt.bar_label(
                i,
                fmt="%.2f",
                label_type="edge",
                fontsize="xx-small",
                padding=-25,
                weight="bold",
                color="white",
            )
        sns.barplot(
            y="connectivity",
            x="dummy_balanced_accuracy",
            data=df_,
            orient="h",
            palette=sns.color_palette("pastel")[0:1],
            order=order,
            ci=None,
            facecolor=(0.8, 0.8, 0.8, 1),
        )
        sns.set_context("notebook")
        plt.xlim(0, 1)
        plt.xlabel("Accuracy")
        plt.ylabel("FC measure")
        fig = plt.gcf()
        if how_many == "three":
            fig.set_size_inches(5, 2)
        if clas == "Subjects":
            plt.title("Across tasks")
        plt.savefig(
            os.path.join(output_dir, f"{clas}_classification_{how_many}.png"),
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(output_dir, f"{clas}_classification_{how_many}.svg"),
            bbox_inches="tight",
        )
        plt.close("all")


### within or binary task classification accuracies ###
cache = DATA_ROOT = "/storage/store/work/haggarwa/"
DATA_ROOT2 = "/storage/store2/work/haggarwa/"
n_parcels = 200
if n_parcels == 400:
    results_dir = "fc_classification_20231013-135722"  # 400 parcels results
    results_pkl = os.path.join(
        DATA_ROOT, results_dir, "all_results.pkl"
    )  # 400 parcels results
    output_dir = "within_task_classification_plots"  # 400 parcels results
elif n_parcels == 200:
    results_dir = "fc_classification_20231115-154004"  # 200 parcels results
    results_pkl = os.path.join(
        DATA_ROOT2, results_dir, "all_results.pkl"
    )  # 200 parcels results
    output_dir = "within_task_classification_plots_200"  # 200 parcels results
output_dir = os.path.join(DATA_ROOT2, output_dir)  # 400 parcels results
os.makedirs(output_dir, exist_ok=True)
df = pd.read_pickle(results_pkl)
# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
# what to classify
classify = ["Tasks", "Subjects", "Runs"]
# tasks
tasks = ["RestingState", "Raiders", "GoodBadUgly", "MonkeyKingdom", "Mario"]
movies = tasks[1:4]

for clas in classify:
    df_ = df[df["classes"] == clas]
    df_.reset_index(inplace=True, drop=True)
    if n_parcels == 400:
        df_.drop(columns=["weights"], inplace=True)
    balanced_accuracies = []
    dummy_balanced_accuracies = []
    for _, row in df_.iterrows():
        balanced_accuracy = balanced_accuracy_score(
            row["true_class"], row["LinearSVC_predicted_class"]
        )
        dummy_balanced_accuracy = balanced_accuracy_score(
            row["true_class"], row["Dummy_predicted_class"]
        )
        balanced_accuracies.append(balanced_accuracy)
        dummy_balanced_accuracies.append(dummy_balanced_accuracy)
    df_["balanced_accuracy"] = balanced_accuracies
    df_["dummy_balanced_accuracy"] = dummy_balanced_accuracies

    if clas == "Runs":
        df_ = df_[df_["task_label"].isin(movies)]

    for how_many in ["all", "three"]:
        if how_many == "all":
            order = [
                "Unregularized correlation",
                "Unregularized partial correlation",
                "Ledoit-Wolf correlation",
                "Ledoit-Wolf partial correlation",
                "Graphical-Lasso correlation",
                "Graphical-Lasso partial correlation",
            ]
        elif how_many == "three":
            order = [
                "Unregularized correlation",
                "Ledoit-Wolf correlation",
                "Graphical-Lasso partial correlation",
            ]
        if clas == "Tasks":
            eval_metric1 = "LinearSVC_auc"
            eval_metric2 = "Dummy_auc"
            legend_cutoff = 9
            palette_init = 1
            xlabel = "AUC"
            title = ""
        elif clas == "Runs":
            eval_metric1 = "balanced_accuracy"
            eval_metric2 = "dummy_balanced_accuracy"
            legend_cutoff = 2
            palette_init = 1
            xlabel = "Accuracy"
            title = ""
        else:
            eval_metric1 = "balanced_accuracy"
            eval_metric2 = "dummy_balanced_accuracy"
            legend_cutoff = 4
            palette_init = 0
            xlabel = "Accuracy"
            title = "Within task"
        sns.set_context("notebook")
        ax_score = sns.barplot(
            y="connectivity",
            x=eval_metric1,
            data=df_,
            orient="h",
            hue="task_label",
            palette=sns.color_palette()[palette_init:],
            order=order,
            # errwidth=1,
        )
        for i in ax_score.containers:
            plt.bar_label(
                i,
                fmt="%.2f",
                label_type="edge",
                fontsize="xx-small",
                padding=-25,
                weight="bold",
                color="white",
            )
        ax_chance = sns.barplot(
            y="connectivity",
            x=eval_metric2,
            data=df_,
            orient="h",
            hue="task_label",
            palette=sns.color_palette("pastel")[palette_init:],
            order=order,
            ci=None,
        )
        # for i in ax_chance.containers:
        #     plt.bar_label(i, fmt="%.2f", label_type="center")
        plt.xlim(0, 1.05)
        plt.xlabel(xlabel)
        plt.ylabel("FC measure")
        plt.title(title)
        legend = plt.legend(
            framealpha=0, loc="center left", bbox_to_anchor=(1, 0.5)
        )
        # remove legend repetition for chance level
        for i, (text, handle) in enumerate(
            zip(legend.texts, legend.legend_handles)
        ):
            if i > legend_cutoff:
                text.set_visible(False)
                handle.set_visible(False)
        legend.set_title("Task")
        fig = plt.gcf()
        if clas == "Tasks":
            fig.set_size_inches(5, 10)
        else:
            fig.set_size_inches(5, 5)
        plt.savefig(
            os.path.join(output_dir, f"{clas}_classification_{how_many}.png"),
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(output_dir, f"{clas}_classification_{how_many}.svg"),
            bbox_inches="tight",
        )
        plt.close()


### fc-sc similarity, sub-spec ###


def insert_stats(ax, p_val, data, loc=[], h=0.15, y_offset=0, x_n=3):
    """
    Insert p-values from statistical tests into boxplots.
    """
    max_y = data.max()
    h = h / 100 * max_y
    y_offset = y_offset / 100 * max_y
    x1, x2 = loc[0], loc[1]
    y = max_y + h + y_offset
    ax.plot([y, y], [x1, x2], lw=1, c="0.25")
    if p_val < 0.0001:
        text = "****"
    if p_val < 0.001:
        text = "***"
    elif p_val < 0.01:
        text = "**"
    elif p_val < 0.05:
        text = "*"
    else:
        text = "ns"
    ax.text(
        y + 3.5,
        ((x1 + x2) * 0.5) - 0.15,
        f"{text}",
        ha="center",
        va="bottom",
        color="0.25",
    )
    ax.set_xticks([*range(0, x_n)])
    ax.axis("off")


# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
# what to classify
classify = ["Tasks", "Subjects", "Runs"]
# tasks
tasks = [
    "RestingState",
    "Raiders",
    "GoodBadUgly",
    "MonkeyKingdom",
    "Mario",
]
# load the data
DATA_ROOT = "/storage/store2/work/haggarwa/"
n_parcels = 400
if n_parcels == 400:
    results_dir = "fc_similarity_20231106-125501"
    output_dir = os.path.join(DATA_ROOT, "fc-sc_subspec_plots")
elif n_parcels == 200:
    results_dir = "fc_similarity_20231117-164946"
    output_dir = os.path.join(DATA_ROOT, "fc-sc_subspec_plots_200")

similarity_data = pd.read_pickle(
    os.path.join(DATA_ROOT, results_dir, "results.pkl")
)
# create matrices showing similarity between tasks
os.makedirs(output_dir, exist_ok=True)
similarity_data = similarity_data[similarity_data["task2"] == "SC"]
sns.set_context("notebook")

for centering in similarity_data["centering"].unique():
    for cov in cov_estimators:
        for measure in measures:
            df = similarity_data[
                (similarity_data["centering"] == centering)
                & (similarity_data["measure"] == f"{cov} {measure}")
            ]
            for test in ["t", "mwu"]:
                d = {
                    "Comparison": [],
                    "FC measure": [],
                    "Task vs. SC": [],
                    "Similarity": [],
                }
                p_values = {}
                for _, row in df.iterrows():
                    n_subs = len(row["kept_subjects"])
                    corr = row["matrix"].reshape(n_subs, n_subs)
                    same_sub = np.diagonal(corr, offset=0).tolist()
                    upper_tri = corr[np.triu_indices_from(corr, k=1)].tolist()
                    lower_tri = corr[np.tril_indices_from(corr, k=-1)].tolist()
                    cross_sub = upper_tri + lower_tri
                    d["Comparison"].extend(
                        [f"{row['comparison']} Within Subject"] * len(same_sub)
                        + [f"{row['comparison']} Across Subject"]
                        * len(cross_sub)
                    )
                    d["Task vs. SC"].extend(
                        [f"{row['comparison']}"]
                        * (len(same_sub) + len(cross_sub))
                    )
                    d["FC measure"].extend(
                        [row["measure"]] * (len(same_sub) + len(cross_sub))
                    )
                    d["Similarity"].extend(same_sub + cross_sub)
                    p_values[row["comparison"]] = row[f"p_value_{test}"]
                d = pd.DataFrame(d)
                hue_order = [
                    "RestingState vs. SC Across Subject",
                    "RestingState vs. SC Within Subject",
                    "Raiders vs. SC Across Subject",
                    "Raiders vs. SC Within Subject",
                    "GoodBadUgly vs. SC Across Subject",
                    "GoodBadUgly vs. SC Within Subject",
                    "MonkeyKingdom vs. SC Across Subject",
                    "MonkeyKingdom vs. SC Within Subject",
                    "Mario vs. SC Across Subject",
                    "Mario vs. SC Within Subject",
                ]
                tasks = [
                    "RestingState vs. SC",
                    "Raiders vs. SC",
                    "GoodBadUgly vs. SC",
                    "MonkeyKingdom vs. SC",
                    "Mario vs. SC",
                ]
                color_palette = []
                for i in range(len(tasks)):
                    color1 = sns.color_palette("pastel")[i]
                    color2 = sns.color_palette()[i]
                    color_palette.extend([color1, color2])
                fig = plt.figure()
                ax1 = plt.subplot2grid((1, 15), (0, 0), colspan=12)
                ax2 = plt.subplot2grid((1, 15), (0, -3))
                sns.violinplot(
                    x="Similarity",
                    y="Comparison",
                    order=hue_order,
                    hue="Comparison",
                    hue_order=hue_order,
                    palette=color_palette,
                    orient="h",
                    data=d,
                    ax=ax1,
                    # fliersize=0,
                    split=True,
                    # width=4,
                    # dodge=True,
                    # linewidth=1,
                    inner="quart",
                )
                for i, task in enumerate(tasks):
                    index = abs((i - len(p_values)) - 1)
                    insert_stats(
                        ax2,
                        p_values[task],
                        d["Similarity"],
                        loc=[index + 0.2, index + 0.6],
                        x_n=len(p_values),
                    )
                # ax1.legend(framealpha=0, loc="bottom center")
                ax1.set_yticks(np.arange(0, 10, 2) + 0.5, tasks)
                ax1.set_ylabel("Task vs. SC")
                ax1.set_xlabel("Corrected Similarity")
                if centering == "centered":
                    if n_parcels == 400:
                        ax1.set_xlim(-0.02, 0.02)
                        ax1.set_xticks(np.arange(-0.02, 0.03, 0.01))
                    elif n_parcels == 200:
                        ax1.set_xlim(-0.03, 0.03)
                        ax1.set_xticks(np.arange(-0.03, 0.035, 0.01))

                plt.title(f"{cov} {measure}", loc="right", x=-1, y=1.05)
                plot_file = os.path.join(
                    output_dir,
                    f"{cov}_{measure}_{centering}_{test}_box.svg",
                )
                plot_file2 = os.path.join(
                    output_dir,
                    f"{cov}_{measure}_{centering}_{test}_box.png",
                )
                plt.savefig(plot_file, bbox_inches="tight", transparent=True)
                plt.savefig(plot_file2, bbox_inches="tight", transparent=False)
                plt.close()
