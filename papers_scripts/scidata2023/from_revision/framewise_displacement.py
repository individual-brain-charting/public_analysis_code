# %%
"""Framewise-displacement (FD) calculation.
from: https://gist.github.com/JulianKlug/68ca5379935e0eedb9bdeed5ab03cf3a
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ibc_public.utils_data import (data_parser, get_subject_session,
                                    DERIVATIVES, CONDITIONS)
# %%
# ############################ FUNCTIONS #####################################
def framewise_displacement(motion_params: np.ndarray):
    """Calculate framewise Displacement (FD) as per Power et al., 2012"""
    motion_diff = np.diff(motion_params, axis=0, prepend=0)
    FD = np.sum(np.abs(motion_diff[:, 0:3]) + 50 * np.abs(motion_diff[:, 3:]),
                axis=1)
    return FD
# %%
def framewise_displacement_from_file(in_file: str):
    """Get the motion params from a motion file."""
    head_motion = np.loadtxt(in_file)
    FD = framewise_displacement(head_motion)
    return FD
# %%
def FD_subject(sub:str, task:str, db:pd.DataFrame):
    """Get the framewise displacement for a subject in a single array"""
    db_sub = db[(db['subject'] == sub) & (db['task'].str.contains(task)) & \
                (db['contrast'] == 'motion')]
    if db_sub.empty:
        print(f'No motion files found for {sub} {task}')
        return None
    all_FD = [framewise_displacement_from_file(row['path'])
              for _, row in db_sub.iterrows()]
    sub_FD = np.concatenate(all_FD)
    return sub_FD

# %%
def plot_subs_FD_distribution(all_subs_FD, PTS, task, out_dir=''):
    """Plot the distribution of framewise displacement for all subjects."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=all_subs_FD, orient='v')
    plt.ylabel('Framewise Displacement [mm]')
    plt.xticks(np.arange(len(PTS)), [f"{sub}" for sub in PTS], rotation=45)
    plt.suptitle(f'Framewise Displacement for {task}')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'FD_{task}.png'), dpi=300)
    
# %%
if __name__ == '__main__':
    # ########################### INPUTS #####################################
    cache = mem = '/storage/store3/work/aponcema/IBC_paper3/cache_two'
    sub_num = [1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]
    TASKS = ['ClipsTrn','ClipsVal','Raiders','WedgeAnti','WedgeClock',
             'ContRing','ExpRing']
    sess_names = ["clips1","clips2","clips3","clips4","raiders1","raiders2"]
    sub_path = [os.path.join(DERIVATIVES, 'sub-%02d' % s) for s in sub_num]
    PTS = [os.path.basename(full_path) for full_path in sub_path]
    # %%
    # ############################## RUN #####################################
    db = data_parser(derivatives=DERIVATIVES,subject_list = PTS,
                     task_list=TASKS,conditions=CONDITIONS,)
    # %%
    # Make a sub_db with the sessions for each subject
    subject_sessions = get_subject_session(sess_names)
    sub_sess = {
        sub: sorted(set(ses for s, ses in subject_sessions if s == sub))
        for sub in PTS
    }
    # %%
    new_db_ = [db[(db['subject'] == sub) & (db['session'] == ses)] 
               for sub, ses_list in sub_sess.items() for ses in ses_list]
    new_db = pd.concat(new_db_, ignore_index=True)

    grouped_tasks = ["Clips", "Raiders", "Wedge", "Ring"]
    for task in grouped_tasks:
        all_subs_FD = [FD_subject(sub, task, new_db) for sub in PTS]
        plot_subs_FD_distribution(all_subs_FD, PTS, task, out_dir=mem)
# %%
