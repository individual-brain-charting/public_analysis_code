"""This script creates 2D UMAP representations of IBC and Wim GBU data, to
 assess the covariate shift between the two datasets. Also tries different 
 scaling methods to reduce the covariate shift"""
import umap
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from skimage import exposure
from matplotlib import pyplot as plt
import seaborn as sns


cache = DATA_ROOT = "/storage/store2/work/haggarwa/"
# load connectomes for Wim GBU
wim_connectomes = pd.read_pickle(
    os.path.join(
        DATA_ROOT,
        "wim_connectivity_20240125-104121",
        "connectomes_200_compcorr.pkl",
    )
)

# load connectomes for IBC GBU
IBC_connectomes = pd.read_pickle(
    os.path.join(
        DATA_ROOT,
        "connectomes_200_comprcorr",
    )
)

IBC_connectomes = IBC_connectomes[IBC_connectomes["tasks"] == "GoodBadUgly"]
IBC_connectomes = IBC_connectomes[
    IBC_connectomes["run_labels"].isin(["run-03", "run-04", "run-05"])
]
IBC_connectomes.reset_index(inplace=True, drop=True)
# rename run labels to match across datasets
IBC_connectomes["run_labels"].replace("run-03", "1", inplace=True)
IBC_connectomes["run_labels"].replace("run-04", "2", inplace=True)
IBC_connectomes["run_labels"].replace("run-05", "3", inplace=True)

wim_connectomes["run_labels"].replace("run-01", "1", inplace=True)
wim_connectomes["run_labels"].replace("run-02", "2", inplace=True)
wim_connectomes["run_labels"].replace("run-03", "3", inplace=True)

wim_connectomes["tasks"].replace(
    {"WimGoodBadUgly": "Mantini et al."}, inplace=True
)
IBC_connectomes["tasks"].replace({"GoodBadUgly": "IBC"}, inplace=True)

connectomes = pd.concat([wim_connectomes, IBC_connectomes], ignore_index=True)
connectomes.reset_index(inplace=True, drop=True)
connectomes["Dataset, run"] = (
    connectomes["tasks"] + ", run " + connectomes["run_labels"]
)

# cov estimators
cov_estimators = ["Unregularized", "Ledoit-Wolf", "Graphical-Lasso"]
# connectivity measures for each cov estimator
measures = ["correlation", "partial correlation"]

output_dir = os.path.join(DATA_ROOT, "umap_ibc_wim_gbu_robust")
os.makedirs(output_dir, exist_ok=True)

for cov_estimator in cov_estimators:
    for measure in measures:
        umap_reducer = umap.UMAP(random_state=42, n_components=2)

        # ibc connectomes
        ibc_fc = np.array(
            IBC_connectomes[f"{cov_estimator} {measure}"].tolist()
        )
        ibc_fc = RobustScaler(unit_variance=True).fit_transform(ibc_fc)
        print(ibc_fc.shape)
        # wim connectomes
        wim_fc = np.array(
            wim_connectomes[f"{cov_estimator} {measure}"].tolist()
        )
        wim_fc = RobustScaler(unit_variance=True).fit_transform(wim_fc)
        print(wim_fc.shape)

        # connectomes[f"{cov_estimator} {measure}"] = connectomes[
        #     f"{cov_estimator} {measure}"
        # ].apply(lambda x: exposure.equalize_hist(np.array(x)))

        fc = np.concatenate([ibc_fc, wim_fc], axis=0)
        print(fc.shape)

        fc_umap = umap_reducer.fit_transform(fc)
        fig, ax = plt.subplots()
        connectomes[f"{cov_estimator} {measure} umap 1"] = fc_umap[:, 0]
        connectomes[f"{cov_estimator} {measure} umap 2"] = fc_umap[:, 1]
        sns.scatterplot(
            data=connectomes,
            x=f"{cov_estimator} {measure} umap 1",
            y=f"{cov_estimator} {measure} umap 2",
            hue="Dataset, run",
            palette="Paired",
            ax=ax,
            hue_order=[
                "IBC, run 1",
                "Mantini et al., run 1",
                "IBC, run 2",
                "Mantini et al., run 2",
                "IBC, run 3",
                "Mantini et al., run 3",
            ],
            s=100,
        )
        sns.move_legend(
            ax,
            "upper center",
            ncol=1,
            # frameon=True,
            # shadow=True,
            bbox_to_anchor=(1.2, 1),
        )
        plt.title(f"{cov_estimator} {measure}")

        plt.savefig(
            os.path.join(
                output_dir,
                f"umap_{cov_estimator}_{measure}.svg",
            ),
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(
                output_dir,
                f"umap_{cov_estimator}_{measure}.png",
            ),
            bbox_inches="tight",
        )
        plt.close()
