# %%
# This script generates ./ibc_data/descriptions.json
# which contains descriptions for all tasks, conditions and
# contrasts of the IBC dataset.
# These descriptions are initially documented in other files
# of the repo.

# %%
import json
import warnings

from pathlib import Path

import numpy as np
import pandas as pd

# %%
# Initialise output structure
d = {"tasks": {}}

# %%
# Add descriptions for all tasks
df_tasks = pd.read_csv("./ibc_data/ibc_tasks.tsv", delimiter="\t")

for _, row in df_tasks.iterrows():
    d["tasks"][row["task"]] = {
        "description": str(row["description"]),
        "conditions": {},
        "contrasts": {},
    }

# %%
# Add descriptions for all conditions
df_conditions = pd.read_csv("./ibc_data/ibc_conditions.tsv", delimiter="\t")

missing_task = []
for _, row in df_conditions.iterrows():
    if row["task"] in d["tasks"]:
        d["tasks"][row["task"]]["conditions"][row["condition"]] = {
            "description": str(row["description"])
        }
    else:
        missing_task.append(row["task"])

missing_task = set(missing_task)

# %%
warnings.warn(
    "The following tasks are missing a description "
    "(task names possibly don't match between files for these tasks):\n"
    f"{missing_task}"
)

# %%
# Add descriptions for all constrasts.
# This includes contrast string description
# but also a list of all conditions used to compute this contrast
df_all_contrasts = pd.read_csv("./ibc_data/all_contrasts.tsv", delimiter="\t")

missing_contrasts = []

for task in d["tasks"].keys():
    task_contrasts_filename = Path(f"./ibc_data/contrasts/{task}.csv")
    if task_contrasts_filename.exists():
        df_contrasts = pd.read_csv(
            task_contrasts_filename, delimiter=",", index_col="condition"
        )
        df_contrasts = df_contrasts.T
        conditions = list(df_contrasts.columns)

        for index, row in df_contrasts.iterrows():
            contrast = row.name

            description = None
            tags = []
            selected_descriptions = df_all_contrasts[
                (df_all_contrasts["task"] == task)
                & (df_all_contrasts["contrast"] == contrast)
            ]
            if len(selected_descriptions) > 0:
                description = selected_descriptions.iloc[0]["pretty name"]
                tags = eval(selected_descriptions.iloc[0]["tags"])
                assert isinstance(tags, list) or tags is None
                tags = list(map(lambda x: str(x), tags))
            else:
                missing_contrasts.append((task, contrast))

            d["tasks"][task]["contrasts"][contrast] = {
                "description": str(description),
                "tags": tags,
                "conditions": {},
            }

            for i, condition in enumerate(conditions):
                weight = row[i]
                weight = float(str(weight).replace(",", "."))
                if not np.isnan(weight):
                    d["tasks"][task]["contrasts"][contrast]["conditions"][
                        condition
                    ] = weight

missing_contrasts = set(missing_contrasts)

# %%
warnings.warn(
    "The following contrasts are missing a description "
    "(task / contrast names possibly don't match between files "
    "for these contrasts):\n"
    f"{missing_contrasts}"
)

# %%
# Save dictionary as json file
with open("./ibc_data/descriptions.json", "w") as f:
    json.dump(d, f)

# %%
