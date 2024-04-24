import os
import pandas as pd
import seaborn as sns

sns.set_theme()
sns.set_style("whitegrid")
sns.set_context("talk")


### table of all accuracies ###
cache = DATA_ROOT = "/storage/store/work/haggarwa/"
DATA_ROOT2 = "/storage/store2/work/haggarwa/"
n_parcels = 200

if n_parcels == 400:
    acrosstask_results_dir = "fc_acrosstask_classification_400_20240118-143947"
    withintask_results_dir = "fc_withintask_classification_400_20240120-154926"
    output_dir = os.path.join(DATA_ROOT2, "fc_accuracy_tables_compcorr")
elif n_parcels == 200:
    acrosstask_results_dir = "fc_acrosstask_classification_200_20240117-185001"
    withintask_results_dir = "fc_withintask_classification_200_20240118-143124"
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
