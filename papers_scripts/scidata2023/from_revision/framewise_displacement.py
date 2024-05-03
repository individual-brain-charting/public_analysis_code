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

def _handle_missing_data(all_FD):
    """For the cases where a sbject didn't perform a task, fill with NaNs."""
    for task_idx, task_data in enumerate(all_FD):
        for subj_idx, subj_data in enumerate(task_data):
            if subj_data is None:  # Check if subj_data is None
                if subj_idx > 0:
                    all_FD[task_idx][subj_idx] = [np.nan] *\
                        len(all_FD[task_idx][subj_idx - 1])
                else:
                    all_FD[task_idx][subj_idx] = [np.nan] * 300
    return all_FD
# %%
def create_df_for_plotting(all_FD, PTS, grouped_tasks):
    """Create a dataframe for plotting the FD data."""
    all_FD = _handle_missing_data(all_FD)
    plot_data = {'Subject': [], 'Task': [], 'FD': []}
    for task_idx, task_data in enumerate(all_FD):
        for sub_idx, sub_data in enumerate(task_data):
            for fd in sub_data:
                plot_data['Subject'].append(PTS[sub_idx])
                plot_data['Task'].append(grouped_tasks[task_idx])
                plot_data['FD'].append(fd)
    df = pd.DataFrame(plot_data)
    return df

# %%
def plot_subs_FD_distribution(df_plot, out_dir=''):
    """Plot the distribution of framewise displacement for all subjects."""

    #plt.figure(figsize=(10, 7))
    #sns.boxplot(data=df_plot, x='Subject', y='FD', hue='Task')
    plt.figure(figsize=(8, 10))
    sns.boxplot(data=df_plot, x='FD', y='Subject', hue='Task')
    #plt.ylabel('Framewise Displacement [mm]')
    #plt.xlabel(None) 
    #plt.ylim(0, 1.0) # limit to 0.1mm
    plt.xlabel('Framewise Displacement [mm]')
    plt.ylabel(None) 
    plt.xlim(0.0, 1.0)

    plt.tight_layout()
    #plt.savefig(os.path.join(out_dir, f'FD.png'), dpi=300)
    plt.savefig(os.path.join(out_dir, 'FD_hor.png'), dpi=300)
# %%
def subplot_task_FD(df_to_plot, out_dir=''):

    fig, axes = plt.subplots(4, 1, figsize=(9, 12), sharey=True)
    axes = axes.flatten()
    for i, task in enumerate(df_to_plot['Task'].unique()):
        task_data = df_to_plot[df_to_plot['Task'] == task]
        sns.boxplot(data=task_data, x='Subject', y='FD', ax=axes[i], 
                     hue='Subject', palette='Set1', legend=False)
        axes[i].set_title(task)
        axes[i].set_xlabel(None)
        axes[i].set_ylabel('Framewise Displacement [mm]')
        axes[i].set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'FD_subplot.png'), dpi=300)

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

    all_FD = []
    grouped_tasks = ["Clips", "Raiders", "Wedge", "Ring"]
    for task in grouped_tasks:
        all_subs_FD = [FD_subject(sub, task, new_db) for sub in PTS]
        all_FD.append(all_subs_FD)
    
    # %%
    df_to_plot = create_df_for_plotting(all_FD, PTS, grouped_tasks)

    # plot all the data in a single plot
    # plot_subs_FD_distribution(df_to_plot, out_dir=cache)
    
    # plot each task as subplots
    subplot_task_FD(df_to_plot, out_dir=cache)
    # %%
