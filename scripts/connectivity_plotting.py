import os
from glob import glob
import pandas as pd
import seaborn as sns
from ibc_public import utils_connectivity as fc
from nilearn.connectome import sym_matrix_to_vec
import numpy as np
from nilearn.connectome import vec_to_sym_matrix
from nilearn.plotting import plot_connectome, view_connectome, view_img_on_surf
from nilearn import image
import matplotlib.pyplot as plt
from nilearn import datasets
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import LinearSVC
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu
import textwrap
import matplotlib as mpl

sns.set_theme()
sns.set_style("whitegrid")
sns.set_context("talk")


def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_yticklabels():
        text = label.get_text()
        labels.append(
            textwrap.fill(text, width=width, break_long_words=break_long_words)
        )
    ax.set_yticklabels(labels, rotation=0)


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
    fontsize=20,
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
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=fontsize)
        ax.set_yticklabels(labels, rotation=0, fontsize=fontsize)
    elif labels is not None and ticks is not None:
        ax.set_xticks(ticks, labels, fontsize=fontsize)
        ax.set_yticks(ticks, labels, fontsize=fontsize)
        ax.tick_params(left=True, bottom=True)
    else:
        ax.set_xticklabels([], fontsize=fontsize)
        ax.set_yticklabels([], fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)

    # save to file
    fig = sns_plot.get_figure()
    fig.savefig(f"{output}.png", bbox_inches="tight")
    fig.savefig(f"{output}.svg", bbox_inches="tight", transparent=True)
    plt.close(fig)


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

### fit classifiers to get weights ###


def get_clas_cov_measure(classify, cov_estimators, measures):
    for clas in classify:
        for cov in cov_estimators:
            for measure in measures:
                yield clas, cov, measure


def fit_classifier(clas, cov, measure, func_data, output_dir):
    weights_file = os.path.join(
        output_dir, f"{clas}_{cov} {measure}_weights.npy"
    )
    if os.path.exists(weights_file):
        print(f"skipping {cov} {measure}, already done")
        pass
    else:
        if clas == "Tasks":
            classes = func_data["tasks"].to_numpy(dtype=object)
        elif clas == "Subjects":
            classes = func_data["subject_ids"].to_numpy(dtype=object)
        elif clas == "Runs":
            func_data["run_task"] = (
                func_data["run_labels"] + "_" + func_data["tasks"]
            )
            classes = func_data["run_task"].to_numpy(dtype=object)
        data = np.array(func_data[f"{cov} {measure}"].values.tolist())
        classifier = LinearSVC(max_iter=100000, dual="auto").fit(data, classes)
        np.save(weights_file, classifier.coef_)


DATA_ROOT = cache = "/storage/store2/work/haggarwa/"
n_parcels = 400
if n_parcels == 400:
    func_data_path = os.path.join(cache, "connectomes_400_comprcorr")
    output_dir = os.path.join(DATA_ROOT, "weights_compcorr")
elif n_parcels == 200:
    func_data_path = os.path.join(cache, "connectomes_200_comprcorr")
    output_dir = os.path.join(DATA_ROOT, "weights_200_compcorr")
os.makedirs(output_dir, exist_ok=True)
func_data = pd.read_pickle(func_data_path)
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
measures = ["correlation", "partial correlation"]
classify = ["Tasks", "Subjects", "Runs"]
x = Parallel(n_jobs=20, verbose=11)(
    delayed(fit_classifier)(
        clas, cov, measure, func_data, output_dir=output_dir
    )
    for clas, cov, measure in get_clas_cov_measure(
        classify, cov_estimators, measures
    )
)

### network pair SVC weight matrices ###


def get_clas_cov_measure(classify, cov_estimators, measures):
    for clas in classify:
        for cov in cov_estimators:
            for measure in measures:
                yield clas, f"{cov} {measure}"


def get_network_labels(atlas):
    networks = atlas["labels"].astype("U")
    hemi_network_labels = []
    network_labels = []
    rename_networks = {
        "Vis": "Visual",
        "Cont": "FrontPar",
        "SalVentAttn": "VentAttn",
    }
    for network in networks:
        components = network.split("_")
        components[2] = rename_networks.get(components[2], components[2])
        hemi_network = " ".join(components[1:3])
        hemi_network_labels.append(hemi_network)
        network_labels.append(components[2])

    return hemi_network_labels, network_labels


def load_transform_weights(clas, cov_measure, transform):
    try:
        weights = np.load(
            os.path.join(weight_dir, f"{clas}_{cov_measure}_weights.npy")
        )
    except FileNotFoundError:
        print(f"skipping {clas} {cov_measure}")
        pass

    if transform == "maxratio":
        weights = np.abs(weights)
        max_weights = np.max(weights, axis=0)
        mask = weights.max(axis=0, keepdims=True) == weights
        mean_other_values = np.mean(weights[~mask], axis=0)
        weights = max_weights / mean_other_values
    elif transform == "l2":
        weights = np.sqrt(np.sum(weights**2, axis=0))

    weight_mat = vec_to_sym_matrix(weights, diagonal=np.ones(n_parcels))

    return weight_mat


def average_over_networks(
    encoded_labels, unique_labels, clas, cov_measure, transform
):
    network_pair_weights = np.zeros((len(unique_labels), len(unique_labels)))
    # get network pair weights
    for network_i in unique_labels:
        index_i = np.where(encoded_labels == network_i)[0]
        for network_j in unique_labels:
            index_j = np.where(encoded_labels == network_j)[0]
            weight_mat = load_transform_weights(clas, cov_measure, transform)
            weight_mat[np.triu_indices_from(weight_mat)] = np.nan
            network_pair_weight = np.nanmean(
                weight_mat[np.ix_(index_i, index_j)]
            )
            network_pair_weights[network_i][network_j] = network_pair_weight

    return network_pair_weights


def plot_network_weight_matrix(
    clas,
    cov_measure,
    atlas,
    output_dir,
    labels_fmt="hemi network",
    transform="maxratio",
    fontsize=20,
):
    if labels_fmt == "hemi network":
        labels = get_network_labels(atlas)[0]
    elif labels_fmt == "network":
        labels = get_network_labels(atlas)[1]

    le = preprocessing.LabelEncoder()
    encoded_labels = le.fit_transform(labels)

    unique_labels = np.unique(encoded_labels)
    network_pair_weights = average_over_networks(
        encoded_labels, unique_labels, clas, cov_measure, transform
    )
    # plot network wise average weights
    get_lower_tri_heatmap(
        network_pair_weights,
        figsize=(5, 5),
        cmap="viridis",
        labels=le.inverse_transform(unique_labels),
        output=os.path.join(
            output_dir, f"{clas}_{cov_measure}_network_weights"
        ),
        triu=True,
        title=cov_measure,
        fontsize=fontsize,
    )


def plot_full_weight_matrix(
    clas,
    cov_measure,
    atlas,
    output_dir,
    transform="maxratio",
    fontsize=20,
):
    weight_mat = load_transform_weights(clas, cov_measure, transform=transform)

    hemi_network_labels = get_network_labels(atlas)[0]

    # get tick locations
    ticks = []
    for i, label in enumerate(hemi_network_labels):
        if label != hemi_network_labels[i - 1]:
            ticks.append(i)

    # keep unique labels
    _, uniq_idx = np.unique(hemi_network_labels, return_index=True)
    hemi_network_labels = np.array(hemi_network_labels)[np.sort(uniq_idx)]

    # plot full matrix
    get_lower_tri_heatmap(
        weight_mat,
        cmap="viridis",
        labels=hemi_network_labels,
        output=os.path.join(output_dir, f"{clas}_{cov_measure}_all_weights"),
        triu=True,
        diag=True,
        title=cov_measure,
        ticks=ticks,
        grid=True,
        fontsize=fontsize,
    )


DATA_ROOT = cache = "/storage/store2/work/haggarwa/"
n_parcels = 400
if n_parcels == 400:
    weight_dir = os.path.join(DATA_ROOT, "weights_compcorr")
    output_dir = os.path.join(DATA_ROOT, "weight_plots_compcorr")
elif n_parcels == 200:
    weight_dir = os.path.join(DATA_ROOT, "weights_200_compcorr")
    output_dir = os.path.join(DATA_ROOT, "weight_plots_200_compcorr")
os.makedirs(output_dir, exist_ok=True)
# get atlas for yeo network labels
atlas = datasets.fetch_atlas_schaefer_2018(
    data_dir=cache, resolution_mm=2, n_rois=n_parcels
)
# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
# what to classify
classify = ["Tasks", "Subjects", "Runs"]

x = Parallel(n_jobs=20, verbose=11)(
    delayed(plot_full_weight_matrix)(
        clas,
        cov_measure,
        atlas,
        transform="l2",
        output_dir=output_dir,
        fontsize=15,
    )
    for clas, cov_measure in get_clas_cov_measure(
        classify, cov_estimators, measures
    )
)
x = Parallel(n_jobs=20, verbose=11)(
    delayed(plot_network_weight_matrix)(
        clas,
        cov_measure,
        atlas,
        labels_fmt="network",
        transform="l2",
        output_dir=output_dir,
        fontsize=15,
    )
    for clas, cov_measure in get_clas_cov_measure(
        classify, cov_estimators, measures
    )
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
    # results_dir = "fc_similarity_20231106-125501"
    # results_dir = "fc_similarity_20240124-162601"  # with compcorr
    results_dir = "fc_similarity_20240411-155121"  # adding back the overall mean similarity
elif n_parcels == 200:
    # results_dir = "fc_similarity_20231117-164946"
    # results_dir = "fc_similarity_20240124-163818"  # with compcorr
    results_dir = "fc_similarity_20240411-155035"  # adding back the overall mean similarity
similarity_data = pd.read_pickle(
    os.path.join(DATA_ROOT, results_dir, "results.pkl")
)
# create matrices showing similarity between tasks
if n_parcels == 400:
    output_dir = os.path.join(DATA_ROOT, "similarity_plots_compcorr")
elif n_parcels == 200:
    output_dir = os.path.join(DATA_ROOT, "similarity_plots_200_compcorr")
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
        with sns.plotting_context("notebook"):
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
                fontsize=15,
            )
# create matrices showing subject-specificity of tasks
if n_parcels == 400:
    output_dir = os.path.join(DATA_ROOT, "subspec_plots_compcorr")
elif n_parcels == 200:
    output_dir = os.path.join(DATA_ROOT, "subspec_plots_200_compcorr")
os.makedirs(output_dir, exist_ok=True)
diffss = []
simss = []
for cov in cov_estimators:
    for measure in measures:
        similarity_values = np.zeros((len(tasks), len(tasks)))
        similarity_type = np.zeros((len(tasks), len(tasks)), dtype=object)
        diffs = []
        sims = []
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
                        diffs.append(same_sub - cross_sub)
                        sims.extend(matrix.flatten())
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
            # similarity_annot[1][3] = "Within\nsubs"
            # similarity_annot[3][1] = "Across\nsubs"
            with sns.plotting_context("notebook"):
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
                    fontsize=15,
                )
            if centering == "centered":
                print(
                    f"{cov} {measure}: {np.mean(diffs):.2f} +/- {np.std(diffs):.2f}"
                )
                print(f"Av sim {cov} {measure}: {np.nanmean(sims):.2f}")
                if (
                    cov == "Graphical-Lasso"
                    and measure == "partial correlation"
                ):
                    diffss.append(diffs)
                    simss.append(sims)
                elif cov == "Ledoit-Wolf" and measure == "correlation":
                    diffss.append(diffs)
                    simss.append(sims)
                elif cov == "Unregularized" and measure == "correlation":
                    diffss.append(diffs)
                    simss.append(sims)
                else:
                    continue
print(
    "\n\n***Testing whether difference between within sub similarity and across sub similarity\n"
    "is greater in Graphical-Lasso partial corr than corr measures***"
)
print(
    "Graphical-Lasso partial corr > Ledoit-Wolf corr\n",
    mannwhitneyu(
        diffss[0],
        diffss[1],
        alternative="greater",
    ),
)
print(
    "Graphical-Lasso partial corr > Unregularized corr\n",
    mannwhitneyu(
        diffss[0],
        diffss[2],
        alternative="greater",
    ),
)
print(
    "\n\n***Testing whether similarity values for Graphical-Lasso partial corr are greater than for corr measures***"
)
print(
    "Graphical-Lasso partial corr > Ledoit-Wolf corr\n",
    mannwhitneyu(
        simss[0],
        simss[1],
        alternative="greater",
    ),
)
print(
    "Graphical-Lasso partial corr > Unregularized corr\n",
    mannwhitneyu(
        simss[0],
        simss[2],
        alternative="greater",
    ),
)
# comparison between distributions of fc-fc similarity values for 400 vs. 200 parcels
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
# colors
colors = ["red", "blue"]
# output directory
output_dir = os.path.join(
    DATA_ROOT, "200V400_fcfc_similarity_comparison_compcorr"
)
os.makedirs(output_dir, exist_ok=True)

for cov in cov_estimators:
    for measure in measures:
        fig, ax = plt.subplots(figsize=(5, 5))
        means = []
        values = []
        for n_parcels, color in zip([200, 400], colors):
            if n_parcels == 400:
                # results_dir = "fc_similarity_20231106-125501"
                # results_dir = "fc_similarity_20240124-162601"  # with compcorr
                results_dir = "fc_similarity_20240411-155121"  # adding back the overall mean similarity
            elif n_parcels == 200:
                # results_dir = "fc_similarity_20231117-164946"
                # results_dir = "fc_similarity_20240124-163818"  # with compcorr
                results_dir = "fc_similarity_20240411-155035"  # adding back the overall mean similarity
            similarity_data = pd.read_pickle(
                os.path.join(DATA_ROOT, results_dir, "results.pkl")
            )
            mask = (similarity_data["measure"] == f"{cov} {measure}") & (
                similarity_data["centering"] == "centered"
            )
            matrix = similarity_data[mask]["matrix"].to_numpy()[0]
            n_subs = len(similarity_data[mask]["kept_subjects"].to_numpy()[0])
            matrix = matrix.reshape((n_subs, n_subs))
            upper_tri = np.triu(matrix, k=1).flatten()
            lower_tri = np.tril(matrix, k=-1).flatten()
            cross_sub = np.concatenate((upper_tri, lower_tri))
            values.append(cross_sub)
            threshold = np.percentile(cross_sub, 1)
            cross_sub_thresh = cross_sub[(cross_sub > threshold)]
            ax.hist(
                cross_sub_thresh.flatten(),
                bins=50,
                label=f"{n_parcels} regions",
                color=color,
                alpha=0.5,
                density=True,
            )
            means.append(np.mean(cross_sub_thresh.flatten()))
        MWU_test = mannwhitneyu(values[0], values[1], alternative="greater")
        ax.annotate(
            f"MWU test\n200 > 400 regions:\np = {MWU_test[1]:.2e}",
            xy=(0.57, 0.83),
            xycoords="axes fraction",
            bbox={"fc": "0.8"},
            fontsize=12,
        )
        ax.axvline(
            means[0],
            color=colors[0],
            linestyle="--",
        )
        ax.axvline(
            means[1],
            color="k",
            linestyle="--",
            label="mean",
        )
        ax.axvline(
            means[1],
            color=colors[1],
            linestyle="--",
        )
        plt.legend(framealpha=0, loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_title(f"{cov} {measure}")
        plt.xlabel("FC-FC Similarity")
        plt.savefig(
            os.path.join(output_dir, f"{cov}_{measure}_hist.png"),
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(output_dir, f"{cov}_{measure}_hist.svg"),
            transparent=True,
            bbox_inches="tight",
        )
        plt.close()
        print(
            f"Testing whether 200 > 400 for {cov} {measure}\n",
            MWU_test,
        )


### connectivity plots ###
# get atlas for yeo network labels
n_parcels = 400
cache = DATA_ROOT = "/storage/store2/work/haggarwa/"
atlas = datasets.fetch_atlas_schaefer_2018(
    data_dir=cache, resolution_mm=2, n_rois=n_parcels
)
networks = atlas["labels"].astype("U")
hemi_network_labels = []
network_labels = []
rename_networks = {
    "Vis": "Visual",
    "Cont": "FrontPar",
    "SalVentAttn": "VentAttn",
}
for network in networks:
    components = network.split("_")
    components[2] = rename_networks.get(components[2], components[2])
    hemi_network = " ".join(components[1:3])
    hemi_network_labels.append(hemi_network)
    network_labels.append(components[2])
ticks = []
unique_hemi_network_labels = []
for i, label in enumerate(hemi_network_labels):
    if label != hemi_network_labels[i - 1]:
        ticks.append(i)
        unique_hemi_network_labels.append(label)

le = preprocessing.LabelEncoder()
encoded_labels = le.fit_transform(network_labels)
unique_encoded_labels = np.unique(encoded_labels)
# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
# what to classify
classify = ["Tasks", "Subjects", "Runs"]
# tasks
tasks = ["RestingState", "Raiders", "GoodBadUgly", "MonkeyKingdom", "Mario"]
# load the data
if n_parcels == 400:
    fc_data = pd.read_pickle(os.path.join(DATA_ROOT, "connectomes2"))
    coords_file = "Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv"
    mats_dir = os.path.join(DATA_ROOT, "connectivity_matrices")
    brain_dir = os.path.join(DATA_ROOT, "brain_connectivity")
    html_dir = os.path.join(DATA_ROOT, "brain_connectivity_html")
elif n_parcels == 200:
    fc_data = pd.read_pickle(
        os.path.join(DATA_ROOT, "connectomes_200_parcels")
    )
    coords_file = "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv"
    mats_dir = os.path.join(DATA_ROOT, "connectivity_matrices_200")
    brain_dir = os.path.join(DATA_ROOT, "brain_connectivity_200")
    html_dir = os.path.join(DATA_ROOT, "brain_connectivity_html_200")
sns.set_context("notebook", font_scale=1.05)
os.makedirs(mats_dir, exist_ok=True)
os.makedirs(brain_dir, exist_ok=True)
os.makedirs(html_dir, exist_ok=True)
coords = pd.read_csv(
    os.path.join(
        cache,
        "schaefer_2018",
        coords_file,
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
                        diagonal=np.ones(n_parcels),
                    )
                except ValueError as e:
                    print(e)
                    print(f"{sub} {task} {cov} {measure} does not exist")
                    continue
                get_lower_tri_heatmap(
                    matrix,
                    title=f"{sub} {task}",
                    output=os.path.join(
                        mats_dir, f"{sub}_{task}_{cov}_{measure}"
                    ),
                    ticks=ticks,
                    labels=unique_hemi_network_labels,
                    grid=True,
                    diag=True,
                    triu=True,
                )

                # get network wise average connectivity
                network_pair_conns = np.zeros(
                    (len(unique_encoded_labels), len(unique_encoded_labels))
                )
                for network_i in unique_encoded_labels:
                    index_i = np.where(encoded_labels == network_i)[0]
                    for network_j in unique_encoded_labels:
                        index_j = np.where(encoded_labels == network_j)[0]
                        matrix[np.triu_indices_from(matrix)] = np.nan
                        network_pair_conn = np.nanmean(
                            matrix[np.ix_(index_i, index_j)]
                        )
                        network_pair_conns[network_i][
                            network_j
                        ] = network_pair_conn
                # plot network wise average connectivity
                get_lower_tri_heatmap(
                    network_pair_conns,
                    title=f"{sub} {task}",
                    output=os.path.join(
                        mats_dir, f"{sub}_{task}_{cov}_{measure}_networks"
                    ),
                    labels=le.inverse_transform(unique_encoded_labels),
                    triu=True,
                    cmap="hot_r",
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


### mean functional connectivity plots ###
n_parcels = 400
cache = DATA_ROOT = "/storage/store2/work/haggarwa/"
atlas = datasets.fetch_atlas_schaefer_2018(
    data_dir=cache, resolution_mm=2, n_rois=n_parcels
)
networks = atlas["labels"].astype("U")
hemi_network_labels = []
network_labels = []
rename_networks = {
    "Vis": "Visual",
    "Cont": "FrontPar",
    "SalVentAttn": "VentAttn",
}
for network in networks:
    components = network.split("_")
    components[2] = rename_networks.get(components[2], components[2])
    hemi_network = " ".join(components[1:3])
    hemi_network_labels.append(hemi_network)
    network_labels.append(components[2])
ticks = []
unique_hemi_network_labels = []
for i, label in enumerate(hemi_network_labels):
    if label != hemi_network_labels[i - 1]:
        ticks.append(i)
        unique_hemi_network_labels.append(label)

le = preprocessing.LabelEncoder()
encoded_labels = le.fit_transform(network_labels)
unique_encoded_labels = np.unique(encoded_labels)
# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
# what to classify
classify = ["Tasks", "Subjects", "Runs"]
# tasks
tasks = ["RestingState", "Raiders", "GoodBadUgly", "MonkeyKingdom", "Mario"]
# load the data
if n_parcels == 400:
    # fc_data = pd.read_pickle(os.path.join(DATA_ROOT, "connectomes2"))
    fc_data = pd.read_pickle(
        os.path.join(DATA_ROOT, "connectomes_400_comprcorr")
    )  # with compcorr
    coords_file = "Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv"
    mats_dir = os.path.join(DATA_ROOT, "mean_connectivity_matrices_compcorr")
elif n_parcels == 200:
    # fc_data = pd.read_pickle(
    #     os.path.join(DATA_ROOT, "connectomes_200_parcels")
    # )
    fc_data = pd.read_pickle(
        os.path.join(DATA_ROOT, "connectomes_200_comprcorr")
    )  # with compcorr
    coords_file = "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv"
    mats_dir = os.path.join(
        DATA_ROOT, "mean_connectivity_matrices_200_compcorr"
    )
sns.set_context("notebook", font_scale=1.05)
os.makedirs(mats_dir, exist_ok=True)
coords = pd.read_csv(
    os.path.join(
        cache,
        "schaefer_2018",
        coords_file,
    )
)[["R", "A", "S"]].to_numpy()
for cov in cov_estimators:
    for measure in measures:
        try:
            matrix = vec_to_sym_matrix(
                np.mean(
                    np.vstack(list(fc_data[f"{cov} {measure}"])),
                    axis=0,
                ),
                diagonal=np.ones(n_parcels),
            )
        except ValueError as e:
            print(e)
            print(f"{sub} {task} {cov} {measure} does not exist")
            continue
        sns.set_context("notebook")
        get_lower_tri_heatmap(
            matrix,
            title=f"{cov} {measure}",
            output=os.path.join(mats_dir, f"full_{cov}_{measure}"),
            ticks=ticks,
            labels=unique_hemi_network_labels,
            grid=True,
            diag=True,
            triu=True,
        )

        # get network wise average connectivity
        network_pair_conns = np.zeros(
            (len(unique_encoded_labels), len(unique_encoded_labels))
        )
        for network_i in unique_encoded_labels:
            index_i = np.where(encoded_labels == network_i)[0]
            for network_j in unique_encoded_labels:
                index_j = np.where(encoded_labels == network_j)[0]
                matrix[np.triu_indices_from(matrix)] = np.nan
                network_pair_conn = np.nanmean(
                    matrix[np.ix_(index_i, index_j)]
                )
                network_pair_conns[network_i][network_j] = network_pair_conn
        # plot network wise average connectivity
        get_lower_tri_heatmap(
            network_pair_conns,
            figsize=(5, 5),
            title=f"{cov} {measure}",
            output=os.path.join(mats_dir, f"networks_{cov}_{measure}"),
            labels=le.inverse_transform(unique_encoded_labels),
            triu=True,
            cmap="viridis",
        )
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

n_parcels = 400
if n_parcels == 400:
    results_dir = "fc_acrosstask_classification_400_20240118-143947"  # with compcorr and fixed resting state confounds
    # results_dir = "fc_acrosstask_classification_400_20231220-101438"  # 400 parcels results
    # results_dir = "fc_acrosstask_classification_400_20231215-175012"  # with stratifiedgroupkfold cross validation
    results_pkl = os.path.join(
        DATA_ROOT2, results_dir, "all_results.pkl"
    )  # 400 parcels results
    output_dir = "classification_plots_compcorr"  # 400 parcels results
elif n_parcels == 200:
    # results_dir = "fc_classification_20231115-140922"  # 200 parcels results
    results_dir = "fc_acrosstask_classification_200_20240117-185001"  # with compcorr and fixed resting state confounds
    results_pkl = os.path.join(
        DATA_ROOT2, results_dir, "all_results.pkl"
    )  # 200 parcels results
    output_dir = "classification_plots_200_compcorr"  # 200 parcels results
output_dir = os.path.join(DATA_ROOT2, output_dir)
os.makedirs(output_dir, exist_ok=True)
df = pd.read_pickle(results_pkl)

# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
# what to classify
classify = ["Tasks", "Subjects", "Runs"]

df[df.select_dtypes(include=["number"]).columns] *= 100

for clas in classify:
    df_ = df[df["classes"] == clas]
    df_.reset_index(inplace=True, drop=True)
    # if n_parcels == 400:
    # df_.drop(columns=["weights"], inplace=True)

    # balanced_accuracies = []
    # dummy_balanced_accuracies = []
    # for _, row in df_.iterrows():
    #     balanced_accuracy = balanced_accuracy_score(
    #         row["true_class"], row["LinearSVC_predicted_class"]
    #     )
    #     dummy_balanced_accuracy = balanced_accuracy_score(
    #         row["true_class"], row["Dummy_predicted_class"]
    #     )
    #     balanced_accuracies.append(balanced_accuracy)
    #     dummy_balanced_accuracies.append(dummy_balanced_accuracy)
    # df_["balanced_accuracy"] = balanced_accuracies
    # df_["dummy_balanced_accuracy"] = dummy_balanced_accuracies

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
                fmt="%.1f",
                label_type="edge",
                fontsize="x-small",
                padding=-45,
                weight="bold",
                color="white",
            )
        wrap_labels(ax_score, 20)
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
        plt.xlabel("Accuracy")
        plt.ylabel("FC measure")
        fig = plt.gcf()
        if how_many == "three":
            fig.set_size_inches(6, 2.5)
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
hatches = [None, "X", "\\", "/", "|"] * 8

cache = DATA_ROOT = "/storage/store/work/haggarwa/"
DATA_ROOT2 = "/storage/store2/work/haggarwa/"
n_parcels = 400

do_hatch = False

if n_parcels == 400:
    results_dir = "fc_withintask_classification_400_20240120-154926"  # with compcorr and fixed resting state confounds
    # results_dir = "fc_withintask_classification_400_20231218-120742"  # 400 parcels results
    # results_dir = "fc_withintask_classification_400_20231215-174943"  # with stratifiedgroupkfold cross validation
    results_pkl = os.path.join(
        DATA_ROOT2, results_dir, "all_results.pkl"
    )  # 400 parcels results
    output_dir = (
        "within_task_classification_plots_compcorr"  # 400 parcels results
    )
elif n_parcels == 200:
    # results_dir = "fc_classification_20231115-154004"  # 200 parcels results
    results_dir = "fc_withintask_classification_200_20240118-143124"  # with compcorr and fixed resting state confounds
    results_pkl = os.path.join(
        DATA_ROOT2, results_dir, "all_results.pkl"
    )  # 200 parcels results
    # output_dir = "within_task_classification_plots_200"  # 200 parcels results
    output_dir = "within_task_classification_plots_200_compcorr"  # with compcorr and fixed resting state confounds
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

df[df.select_dtypes(include=["number"]).columns] *= 100

for clas in classify:
    df_ = df[df["classes"] == clas]
    df_.reset_index(inplace=True, drop=True)
    # if n_parcels == 400:
    # df_.drop(columns=["weights"], inplace=True)
    # balanced_accuracies = []
    # dummy_balanced_accuracies = []
    # for _, row in df_.iterrows():
    #     balanced_accuracy = balanced_accuracy_score(
    #         row["true_class"], row["LinearSVC_predicted_class"]
    #     )
    #     dummy_balanced_accuracy = balanced_accuracy_score(
    #         row["true_class"], row["Dummy_predicted_class"]
    #     )
    #     balanced_accuracies.append(balanced_accuracy)
    #     dummy_balanced_accuracies.append(dummy_balanced_accuracy)
    # df_["balanced_accuracy"] = balanced_accuracies
    # df_["dummy_balanced_accuracy"] = dummy_balanced_accuracies
    if clas == "Runs":
        print(len(df_))
        print(df_["task_label"].unique())
        df_ = df_[df_["task_label"].isin(movies)]
        print(len(df_))
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
            legend_cutoff = 10
            palette_init = 1
            xlabel = "AUC"
            title = ""
            rest_colors = sns.color_palette("tab20c")[0:4]
            movie_colors = sns.color_palette("tab20c")[4:7]
            mario_colors = sns.color_palette("tab20c")[8:11]
            color_palette = rest_colors + movie_colors + mario_colors
            bar_label_color = "white"
            bar_label_weight = "bold"
        elif clas == "Runs":
            eval_metric1 = "balanced_accuracy"
            eval_metric2 = "dummy_balanced_accuracy"
            legend_cutoff = 3
            palette_init = 1
            xlabel = "Accuracy"
            title = ""
            movie_colors = sns.color_palette("tab20c")[4:7]
            color_palette = movie_colors
            bar_label_color = "white"
            bar_label_weight = "bold"
        else:
            eval_metric1 = "balanced_accuracy"
            eval_metric2 = "dummy_balanced_accuracy"
            legend_cutoff = 5
            palette_init = 0
            xlabel = "Accuracy"
            title = ""
            rest_colors = sns.color_palette("tab20c")[0]
            movie_colors = sns.color_palette("tab20c")[4:7]
            mario_colors = sns.color_palette("tab20c")[8]
            color_palette = [rest_colors] + movie_colors + [mario_colors]
            bar_label_color = "white"
            bar_label_weight = "bold"
        ax_score = sns.barplot(
            y="connectivity",
            x=eval_metric1,
            data=df_,
            orient="h",
            hue="task_label",
            palette=color_palette,
            order=order,
            # errwidth=1,
        )
        wrap_labels(ax_score, 20)
        for i, container in enumerate(ax_score.containers):
            plt.bar_label(
                container,
                fmt="%.1f",
                label_type="edge",
                fontsize="x-small",
                padding=-45,
                weight=bar_label_weight,
                color=bar_label_color,
            )
            if do_hatch:
                # Loop over the bars
                for thisbar in container.patches:
                    # Set a different hatch for each bar
                    thisbar.set_hatch(hatches[i])
        ax_chance = sns.barplot(
            y="connectivity",
            x=eval_metric2,
            data=df_,
            orient="h",
            hue="task_label",
            palette=sns.color_palette("pastel")[7:],
            order=order,
            errorbar=None,
            facecolor=(0.8, 0.8, 0.8, 1),
        )
        # for i in ax_chance.containers:
        #     plt.bar_label(i, fmt="%.2f", label_type="center")
        # plt.xlim(0, 105)

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
            else:
                if do_hatch:
                    handle._hatch = hatches[i]

            if i == legend_cutoff:
                text.set_text("Chance-level")

        legend.set_title("Task")
        fig = plt.gcf()
        if clas == "Tasks":
            fig.set_size_inches(6, 10)
        else:
            fig.set_size_inches(6, 6)

        plt.savefig(
            os.path.join(output_dir, f"{clas}_classification_{how_many}.png"),
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(output_dir, f"{clas}_classification_{how_many}.svg"),
            bbox_inches="tight",
            transparent=True,
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
    ax.plot([y, y], [x1, x2], lw=2, c="0.25")
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


hatches = [None, "X", "\\", "/", "|"] * 8
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
n_parcels = 200
if n_parcels == 400:
    # results_dir = "fc_similarity_20231106-125501"
    # results_dir = "fc_similarity_20240124-162601"  # with compcorr
    results_dir = "fc_similarity_20240411-155121"  # adding back the overall mean similarity
    output_dir = os.path.join(DATA_ROOT, "fc-sc_subspec_plots_compcorr")
elif n_parcels == 200:
    # results_dir = "fc_similarity_20231117-164946"
    # results_dir = "fc_similarity_20240124-163818"  # with compcorr
    results_dir = "fc_similarity_20240411-155035"  # adding back the overall mean similarity
    output_dir = os.path.join(DATA_ROOT, "fc-sc_subspec_plots_200_compcorr")

similarity_data = pd.read_pickle(
    os.path.join(DATA_ROOT, results_dir, "results.pkl")
)
os.makedirs(output_dir, exist_ok=True)
similarity_data = similarity_data[similarity_data["task2"] == "SC"]

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
                # color_palette = []
                # for i in range(len(tasks)):
                #     color1 = sns.color_palette("pastel")[i]
                #     color2 = sns.color_palette()[i]
                #     color_palette.extend([color1, color2])
                rest_colors = sns.color_palette("tab20c")[0]
                movie_1 = sns.color_palette("tab20c")[4]
                movie_2 = sns.color_palette("tab20c")[5]
                movie_3 = sns.color_palette("tab20c")[6]
                mario_colors = sns.color_palette("tab20c")[8]
                color_palette = list(
                    [rest_colors] * 2
                    + [movie_1] * 2
                    + [movie_2] * 2
                    + [movie_3] * 2
                    + [mario_colors] * 2
                )
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
                    inner=None,
                )
                violin_patches = []
                for i, patch in enumerate(ax1.get_children()):
                    if isinstance(patch, mpl.collections.PolyCollection):
                        violin_patches.append(patch)
                for i, patch in enumerate(violin_patches):
                    if i % 2 == 0:
                        # Loop over the bars
                        patch.set_hatch(hatches[1])
                        patch.set_edgecolor("k")

                legend_elements = [
                    mpl.patches.Patch(
                        facecolor="white",
                        edgecolor="k",
                        label="Across Subject",
                    ),
                    mpl.patches.Patch(
                        facecolor="white",
                        edgecolor="k",
                        label="Within Subject",
                    ),
                ]
                legend_elements[0].set_hatch(hatches[1])
                ax1.legend(
                    framealpha=0,
                    loc="center left",
                    bbox_to_anchor=(1.2, 0.5),
                    handles=legend_elements,
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
                ax1.set_yticks(np.arange(0, 10, 2) + 0.5, tasks)
                ax1.set_ylabel("Task vs. SC")
                ax1.set_xlabel("Similarity")
                # if centering == "centered":
                #     if n_parcels == 400:
                #         ax1.set_xlim(-0.02, 0.02)
                #         ax1.set_xticks(np.arange(-0.02, 0.03, 0.01))
                #     elif n_parcels == 200:
                #         ax1.set_xlim(-0.03, 0.03)
                #         ax1.set_xticks(np.arange(-0.03, 0.035, 0.01))
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

### fc-sc similarity, barplots ###
hatches = [None, "X", "\\", "/", "|"] * 8
do_hatch = False
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
    # results_dir = "fc_similarity_20231106-125501"
    # results_dir = "fc_similarity_20240124-162601"  # with compcorr
    results_dir = "fc_similarity_20240411-155121"  # adding back the overall mean similarity
    output_dir = os.path.join(DATA_ROOT, "fc-sc_similarity_plots_compcorr")
elif n_parcels == 200:
    # results_dir = "fc_similarity_20231117-164946"
    # results_dir = "fc_similarity_20240124-163818"  # with compcorr
    results_dir = "fc_similarity_20240411-155035"  # adding back the overall mean similarity
    output_dir = os.path.join(DATA_ROOT, "fc-sc_similarity_plots_200_compcorr")

similarity_data = pd.read_pickle(
    os.path.join(DATA_ROOT, results_dir, "results.pkl")
)
os.makedirs(output_dir, exist_ok=True)
similarity_data = similarity_data[similarity_data["task2"] == "SC"]

for centering in similarity_data["centering"].unique():
    df = similarity_data[similarity_data["centering"] == centering]
    for how_many in ["all", "three"]:
        if how_many == "all":
            fc_measure_order = [
                "Unregularized correlation",
                "Unregularized partial correlation",
                "Ledoit-Wolf correlation",
                "Ledoit-Wolf partial correlation",
                "Graphical-Lasso correlation",
                "Graphical-Lasso partial correlation",
            ]
        elif how_many == "three":
            fc_measure_order = [
                "Unregularized correlation",
                "Ledoit-Wolf correlation",
                "Graphical-Lasso partial correlation",
            ]

        d = {"FC measure": [], "Similarity": [], "Comparison": []}
        for _, row in df.iterrows():
            corr = row["matrix"].tolist()
            d["Similarity"].extend(corr)
            d["FC measure"].extend([row["measure"]] * len(corr))
            d["Comparison"].extend([row["comparison"]] * len(corr))
        d = pd.DataFrame(d)
        fig, ax = plt.subplots()

        hue_order = [
            "RestingState vs. SC",
            "Raiders vs. SC",
            "GoodBadUgly vs. SC",
            "MonkeyKingdom vs. SC",
            "Mario vs. SC",
        ]
        name = "fc_sc"
        rest_colors = sns.color_palette("tab20c")[0]
        movie_colors = sns.color_palette("tab20c")[4:7]
        mario_colors = sns.color_palette("tab20c")[8]
        color_palette = [rest_colors] + movie_colors + [mario_colors]
        ax = sns.barplot(
            x="Similarity",
            y="FC measure",
            order=fc_measure_order,
            hue="Comparison",
            orient="h",
            hue_order=hue_order,
            palette=color_palette,
            data=d,
            ax=ax,
            # errorbar=None,
        )
        wrap_labels(ax, 20)
        for i, container in enumerate(ax.containers):
            plt.bar_label(
                container,
                fmt="%.2f",
                label_type="edge",
                fontsize="x-small",
                padding=-45,
                weight="bold",
                color="white",
            )
            if do_hatch:
                # Loop over the bars
                for thisbar in container.patches:
                    # Set a different hatch for each bar
                    thisbar.set_hatch(hatches[i])

        legend = ax.legend(
            framealpha=0, loc="center left", bbox_to_anchor=(1, 0.5)
        )
        if do_hatch:
            for i, handle in enumerate(legend.legend_handles):
                handle._hatch = hatches[i]

        plot_file = os.path.join(
            output_dir,
            f"similarity_{name}_{centering}_{how_many}.svg",
        )
        plot_file2 = os.path.join(
            output_dir,
            f"similarity_{name}_{centering}_{how_many}.png",
        )
        if how_many == "three":
            fig.set_size_inches(5, 5)
        else:
            fig.set_size_inches(5, 10)
        plt.savefig(plot_file, bbox_inches="tight", transparent=True)
        plt.savefig(plot_file2, bbox_inches="tight", transparent=False)
        plt.close()


### fc-sc similarity, network-wise matrix ###
def mean_connectivity(data, tasks, cov_estimators, measures):
    """Average connectivity across runs for each subject and task.

    Parameters
    ----------
    data : pandas dataframe
        a dataframe with flattened connectivity matrices with a
        column for each fc measure (created by joining covariance
        estimator and the measure with a space), a column for
        the task, and a column for the subject
    tasks : list
        a list of tasks to average connectivity across runs
    cov_estimators : list
        a list of covariance estimators
    measures : list
        a list of connectivity measures estimated by each covariance

    Returns
    -------
    pandas dataframe
        a dataframe with the average connectivity for each subject,
        task, and measure in long format
    """
    av_connectivity = []
    for task in tasks:
        task_data = data[data["tasks"] == task]
        task_subjects = task_data["subject_ids"].unique()
        for sub in task_subjects:
            df = task_data[task_data["subject_ids"] == sub]
            for cov in cov_estimators:
                for measure in measures:
                    connectivity = df[cov + " " + measure].tolist()
                    connectivity = np.array(connectivity)
                    connectivity = connectivity.mean(axis=0)
                    av_connectivity.append(
                        {
                            "task": task,
                            "subject": sub,
                            "connectivity": connectivity,
                            "measure": cov + " " + measure,
                        }
                    )

    return pd.DataFrame(av_connectivity)


def get_network_labels(atlas):
    networks = atlas["labels"].astype("U")
    hemi_network_labels = []
    network_labels = []
    rename_networks = {
        "Vis": "Visual",
        "Cont": "FrontPar",
        "SalVentAttn": "VentAttn",
    }
    for network in networks:
        components = network.split("_")
        components[2] = rename_networks.get(components[2], components[2])
        hemi_network = " ".join(components[1:3])
        hemi_network_labels.append(hemi_network)
        network_labels.append(components[2])

    return hemi_network_labels, network_labels


cache = DATA_ROOT = "/storage/store2/work/haggarwa/"

labels_fmt = "network"
n_parcels = 400
if n_parcels == 400:
    fc_data_path = os.path.join(cache, "connectomes_400_comprcorr")
    sc_data_path = os.path.join(cache, "sc_data_native_new")
    output_dir = "fc-sc_similarity_networkwise_plots_compcorr"
elif n_parcels == 200:
    fc_data_path = os.path.join(cache, "connectomes_200_comprcorr")
    sc_data_path = os.path.join(cache, "sc_data_native_200")
    output_dir = "fc-sc_similarity_networkwise_plots_200_compcorr"

output_dir = os.path.join(DATA_ROOT, output_dir)
os.makedirs(output_dir, exist_ok=True)
fc_data = pd.read_pickle(fc_data_path)
sc_data = pd.read_pickle(sc_data_path)

# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
# tasks
tasks = [
    "RestingState",
    "Raiders",
    "GoodBadUgly",
    "MonkeyKingdom",
    "Mario",
]

# get atlas for yeo network labels
atlas = datasets.fetch_atlas_schaefer_2018(
    data_dir=cache, resolution_mm=2, n_rois=n_parcels
)

fc_data = mean_connectivity(fc_data, tasks, cov_estimators, measures)
fc_data.reset_index(drop=True, inplace=True)
sc_data.reset_index(drop=True, inplace=True)

if labels_fmt == "hemi network":
    labels = get_network_labels(atlas)[0]
elif labels_fmt == "network":
    labels = get_network_labels(atlas)[1]

le = preprocessing.LabelEncoder()
encoded_labels = le.fit_transform(labels)

unique_labels = np.unique(encoded_labels)

results = []

for cov in cov_estimators:
    for measure in measures:
        for task in tasks:
            func = fc_data[fc_data["task"] == task]
            func = func[func["measure"] == cov + " " + measure]
            # get func and struc conn for each subject
            for sub in func["subject"].unique():
                sub_func = func[func["subject"] == sub]
                sub_func_mat = vec_to_sym_matrix(
                    sub_func["connectivity"].values[0],
                    diagonal=np.ones(n_parcels),
                )
                sub_func_mat[np.triu_indices_from(sub_func_mat)] = np.nan
                sub_struc = sc_data[sc_data["subject"] == sub]
                sub_struc_mat = vec_to_sym_matrix(
                    sub_struc["connectivity"].values[0],
                    diagonal=np.ones(n_parcels),
                )
                sub_struc_mat[np.triu_indices_from(sub_struc_mat)] = np.nan

                # create empty matrix for network pair correlations
                network_pair_corr = np.zeros(
                    (len(unique_labels), len(unique_labels))
                )
                print(f"\n\n{task} {sub} {cov} {measure}\n\n")
                # get the nodes indices for each network
                for network_i in unique_labels:
                    index_i = np.where(encoded_labels == network_i)[0]
                    # print(index_i)
                    for network_j in unique_labels:
                        index_j = np.where(encoded_labels == network_j)[0]
                        # print(index_j)
                        # func connectivity for network pair
                        sub_func_network = sub_func_mat[
                            np.ix_(index_i, index_j)
                        ]
                        sub_func_network = sub_func_network[
                            ~np.isnan(sub_func_network)
                        ].flatten()
                        # print(sub_func_network)
                        # struc connectivity for network pair
                        sub_struc_network = sub_struc_mat[
                            np.ix_(index_i, index_j)
                        ]
                        sub_struc_network = sub_struc_network[
                            ~np.isnan(sub_struc_network)
                        ].flatten()
                        # print(sub_struc_network)
                        # correlation between func and struc connectivity
                        corr = np.corrcoef(sub_struc_network, sub_func_network)
                        print(corr, f"{task} {sub} {cov} {measure}")
                        network_pair_corr[network_i][network_j] = corr[0][1]
                result = {
                    "corr": network_pair_corr,
                    "task": task,
                    "subject": sub,
                    "cov measure": cov + " " + measure,
                }
                results.append(result)

results = pd.DataFrame(results)

for _, row in results.iterrows():
    task = row["task"]
    sub = row["subject"]
    cov = row["cov measure"]
    corr = row["corr"]
    title = f"{task} {sub} {cov}"
    output = os.path.join(output_dir, f"{task}_{sub}_{cov}")
    get_lower_tri_heatmap(
        corr,
        figsize=(5, 5),
        cmap="hot_r",
        title=title,
        labels=le.inverse_transform(unique_labels),
        output=output,
        triu=True,
    )

# take a mean of the network wise correlations across subjects
# gives a network wise correlation matrix for each task and cov measure
fc_sc_corr_tasks = (
    results.groupby(["task", "cov measure"]).mean().reset_index()
)
for _, row in results.iterrows():
    task = row["task"]
    sub = row["subject"]
    cov = row["cov measure"]
    corr = row["corr"]
    title = f"{task} {sub} {cov}"
    output = os.path.join(output_dir, f"mean_{task}_{cov}")
    get_lower_tri_heatmap(
        corr,
        figsize=(5, 5),
        cmap="hot_r",
        title=title,
        labels=le.inverse_transform(unique_labels),
        output=output,
        triu=True,
    )

# get the mean of the network wise correlations across tasks
# gives a network wise correlation matrix for each cov measure and subject
fc_sc_corr_subjects = (
    results.groupby(["task", "cov measure"]).mean().reset_index()
)
for _, row in fc_sc_corr_subjects.iterrows():
    task = row["task"]
    sub = row["subject"]
    cov = row["cov measure"]
    corr = row["corr"]
    title = f"{task} {sub} {cov}"
    output = os.path.join(output_dir, f"mean_{sub}_{cov}")
    get_lower_tri_heatmap(
        corr,
        figsize=(5, 5),
        cmap="hot_r",
        title=title,
        labels=le.inverse_transform(unique_labels),
        output=output,
        triu=True,
    )

# get the mean of the network wise correlations across tasks and subjects
# gives a network wise correlation matrix for each cov measure
fc_sc_corr = results.groupby(["cov measure"]).mean().reset_index()
for _, row in fc_sc_corr.iterrows():
    task = row["task"]
    sub = row["subject"]
    cov = row["cov measure"]
    corr = row["corr"]
    title = f"{task} {sub} {cov}"
    output = os.path.join(output_dir, f"mean_{cov}")
    get_lower_tri_heatmap(
        corr,
        figsize=(5, 5),
        cmap="hot_r",
        title=title,
        labels=le.inverse_transform(unique_labels),
        output=output,
        triu=True,
    )


### transfer IBC -> Wim connectivity matrices ###
# get atlas for yeo network labels
cache = "/storage/store2/work/haggarwa/"
DATA_ROOT = "/storage/store2/work/haggarwa/"
IBC_ROOT = os.path.join(DATA_ROOT, "ibc_sync_wim_connectivity_20231206-110710")
WIM_ROOT = os.path.join(DATA_ROOT, "wim_connectivity_20231205-142311")

atlas = datasets.fetch_atlas_schaefer_2018(
    data_dir=cache, resolution_mm=2, n_rois=200
)
networks = atlas["labels"].astype("U")
hemi_network_labels = []
for network in networks:
    components = network.split("_")
    hemi_network = "_".join(components[1:3])
    hemi_network_labels.append(hemi_network)
ticks = []
unique_labels = []
for i, label in enumerate(hemi_network_labels):
    if label != hemi_network_labels[i - 1]:
        ticks.append(i)
        unique_labels.append(label)
# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
for dataset in ["ibc", "wim"]:
    if dataset == "ibc":
        # load the data
        fc_data = pd.read_pickle(os.path.join(IBC_ROOT, "connectomes_200.pkl"))
        mats_dir = os.path.join(IBC_ROOT, "connectivity_matrices")
    elif dataset == "wim":
        # load the data
        fc_data = pd.read_pickle(os.path.join(WIM_ROOT, "connectomes_200.pkl"))
        mats_dir = os.path.join(WIM_ROOT, "connectivity_matrices")
    _, uniq_idx = np.unique(hemi_network_labels, return_index=True)
    hemi_network_labels = np.array(hemi_network_labels)[np.sort(uniq_idx)]
    sns.set_context("notebook", font_scale=1.05)
    os.makedirs(mats_dir, exist_ok=True)
    for sub in tqdm(np.unique(fc_data["subject_ids"]), desc=dataset):
        for cov in cov_estimators:
            for measure in measures:
                try:
                    vector = np.mean(
                        np.vstack(
                            list(
                                fc_data[(fc_data["subject_ids"] == sub)][
                                    f"{cov} {measure}"
                                ]
                            )
                        ),
                        axis=0,
                    )
                    matrix = vec_to_sym_matrix(vector, diagonal=np.ones(200))
                except ValueError:
                    print(f"{sub} {cov} {measure} does not exist")
                    continue
                get_lower_tri_heatmap(
                    matrix,
                    title=f"{sub} {dataset}",
                    output=os.path.join(mats_dir, f"{sub}_{cov}_{measure}"),
                    ticks=ticks,
                    labels=hemi_network_labels,
                    grid=True,
                    diag=True,
                    triu=True,
                )
                plt.close("all")

### transfer IBC -> Wim connectivity distribution plots ###
cache = "/storage/store2/work/haggarwa/"
DATA_ROOT = "/storage/store2/work/haggarwa/"
IBC_ROOT = os.path.join(DATA_ROOT, "ibc_sync_wim_connectivity_20231206-110710")
WIM_ROOT = os.path.join(DATA_ROOT, "wim_connectivity_20231205-142311")
atlas = datasets.fetch_atlas_schaefer_2018(
    data_dir=cache, resolution_mm=2, n_rois=200
)
networks = atlas["labels"].astype("U")
hemi_network_labels = []
for network in networks:
    components = network.split("_")
    hemi_network = "_".join(components[1:3])
    hemi_network_labels.append(hemi_network)
ticks = []
unique_labels = []
for i, label in enumerate(hemi_network_labels):
    if label != hemi_network_labels[i - 1]:
        ticks.append(i)
        unique_labels.append(label)
# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]

dist_dir = os.path.join(
    DATA_ROOT, "transfer_ibc_wim_connectivity_distributions"
)
os.makedirs(dist_dir, exist_ok=True)
for cov in cov_estimators:
    for measure in measures:
        fig, ax = plt.subplots()
        for dataset in ["ibc", "wim"]:
            if dataset == "ibc":
                # load the data
                fc_data = pd.read_pickle(
                    os.path.join(IBC_ROOT, "connectomes_200.pkl")
                )
                color = "blue"
            elif dataset == "wim":
                # load the data
                fc_data = pd.read_pickle(
                    os.path.join(WIM_ROOT, "connectomes_200.pkl")
                )
                color = "red"
            try:
                vector = np.mean(
                    np.vstack(list(fc_data[f"{cov} {measure}"])),
                    axis=0,
                )
                matrix = vec_to_sym_matrix(vector, diagonal=np.ones(200))
            except ValueError:
                print(f"{cov} {measure} does not exist")
                continue
            get_lower_tri_heatmap(
                matrix,
                title=f"{dataset} {cov} {measure}",
                output=os.path.join(
                    dist_dir, f"mat_{dataset}_{cov}_{measure}"
                ),
                ticks=ticks,
                labels=unique_labels,
                grid=True,
                diag=True,
                triu=True,
            )
            ax.hist(
                vector,
                bins=500,
                density=True,
                log=True,
                # stacked=True,
                label=dataset,
                color=color,
                alpha=0.5,
            )
            ax.axvline(
                np.mean(vector), linestyle="dashed", linewidth=1, color=color
            )
        ax.legend()
        ax.set_title(f"{cov} {measure}")
        plt.savefig(
            os.path.join(dist_dir, f"dist_{cov}_{measure}.png"),
            bbox_inches="tight",
        )
        plt.close(fig)


### table of all accuracies ###
cache = DATA_ROOT = "/storage/store/work/haggarwa/"
DATA_ROOT2 = "/storage/store2/work/haggarwa/"
n_parcels = 200

if n_parcels == 400:
    # acrosstask_results_dir = "fc_acrosstask_classification_400_20231220-101438"
    # withintask_results_dir = "fc_withintask_classification_400_20231218-120742"
    acrosstask_results_dir = (
        "fc_acrosstask_classification_400_20240118-143947"  # with compcorr
    )
    withintask_results_dir = (
        "fc_withintask_classification_400_20240120-154926"  # with compcorr
    )
    output_dir = os.path.join(DATA_ROOT2, "fc_accuracy_tables_compcorr")
elif n_parcels == 200:
    # acrosstask_results_dir = "fc_acrosstask_classification_200_20240101-205818"  # fc_classification_20231115-140922 if you need accuracy across all tasks and not just movies
    # withintask_results_dir = "fc_classification_20231115-154004"
    acrosstask_results_dir = (
        "fc_acrosstask_classification_200_20240117-185001"  # with compcorr
    )
    withintask_results_dir = (
        "fc_withintask_classification_200_20240118-143124"  # with compcorr
    )
    output_dir = os.path.join(DATA_ROOT2, "fc_accuracy_tables_200_compcorr")
os.makedirs(output_dir, exist_ok=True)
acrosstask_results_pkl = os.path.join(
    DATA_ROOT2, acrosstask_results_dir, "all_results.pkl"
)
withintask_results_pkl = os.path.join(
    DATA_ROOT2, withintask_results_dir, "all_results.pkl"
)

acrosstask_df = pd.read_pickle(acrosstask_results_pkl).reset_index(drop=True)
withintask_df = pd.read_pickle(withintask_results_pkl).reset_index(drop=True)

# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
# what to classify
classify = ["Tasks", "Subjects", "Runs"]
# tasks
tasks = ["RestingState", "Raiders", "GoodBadUgly", "MonkeyKingdom", "Mario"]

# get accuracies for each classification scenario
for clas in classify:
    print(clas)
    classifying_df = pd.concat(
        [
            acrosstask_df[acrosstask_df["classes"] == clas],
            withintask_df[withintask_df["classes"] == clas],
        ]
    )
    classifying_df.reset_index(inplace=True, drop=True)
    for metric in [
        "balanced_accuracy",
        "dummy_balanced_accuracy",
        "LinearSVC_auc",
        "Dummy_auc",
    ]:
        mean_acc = (
            classifying_df.groupby(["task_label", "connectivity"])[metric]
            .mean()
            .round(2)
        )
        mean_acc = mean_acc.unstack(level=1)
        mean_acc["mean"] = mean_acc.mean(axis=1).round(2)
        mean_acc = mean_acc[
            [
                "Unregularized correlation",
                "Unregularized partial correlation",
                "Ledoit-Wolf correlation",
                "Ledoit-Wolf partial correlation",
                "Graphical-Lasso correlation",
                "Graphical-Lasso partial correlation",
                "mean",
            ]
        ]
        mean_acc.to_csv(
            os.path.join(DATA_ROOT2, output_dir, f"{clas}_mean_{metric}.csv")
        )


### plot fmri image for methods
cache = DATA_ROOT = "/storage/store/work/haggarwa/"
DATA_ROOT2 = "/storage/store2/work/haggarwa/"
output_dir = os.path.join(DATA_ROOT2, "fmri_methods")
os.makedirs(output_dir, exist_ok=True)

fmri_file = "/storage/store2/data/ibc/derivatives/sub-04/ses-12/func/wrdcsub-04_ses-12_task-MTTNS_dir-pa_run-01_bold.nii.gz"

mean_fmri = image.mean_img(fmri_file)

view_img_on_surf(mean_fmri, surf_mesh="fsaverage").save_as_html(
    "output_dir/fmri.html"
)


### plot reliability

# cov estimators
cov_estimators = ["Graphical-Lasso", "Ledoit-Wolf", "Unregularized"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]
# tasks
tasks = [
    "RestingState",
    "Raiders",
    "GoodBadUgly",
]
# load the data
DATA_ROOT = "/storage/store2/work/haggarwa/"
n_parcels = 400
results_dir = f"reliability_{n_parcels}"
reliability_data = pd.read_pickle(
    os.path.join(DATA_ROOT, results_dir, f"corrs_full_mat_{n_parcels}")
)
keep_only = [
    "Unregularized correlation",
    "Ledoit-Wolf correlation",
    "Graphical-Lasso partial correlation",
    "time_series",
]
hue_order = [
    "RestingState",
    "Raiders",
    "GoodBadUgly",
]
rest_colors = [sns.color_palette("tab20c")[0]]
movie_colors = sns.color_palette("tab20c")[4:7]
color_palette = rest_colors + movie_colors
ax = sns.boxplot(
    x="correlation",
    y="measure",
    hue="task",
    data=reliability_data,
    palette=color_palette,
    orient="h",
    order=keep_only,
    hue_order=hue_order,
)
wrap_labels(ax, 20)
legend = ax.legend(framealpha=0, loc="center left", bbox_to_anchor=(1, 0.5))
ax.set_xlabel("Reliability")
ax.set_ylabel("Measure")
plot_file = os.path.join(
    results_dir,
    "reliability.svg",
)
plot_file2 = os.path.join(
    results_dir,
    "reliability.png",
)
plt.savefig(plot_file, bbox_inches="tight", transparent=True)
plt.savefig(plot_file2, bbox_inches="tight", transparent=False)
plt.close()
