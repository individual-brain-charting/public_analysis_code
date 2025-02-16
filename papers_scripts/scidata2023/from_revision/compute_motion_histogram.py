"""
Compute the motion histogram across sessions and participants for tasks in
the 3rd release
"""
# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from ibc_public.utils_data import (data_parser, get_subject_session,
                                    DERIVATIVES, CONDITIONS)
# %%
def motion_histogram(db):
    """compute motion histograms from realignment files"""
    rps = list(db[db.contrast == 'motion'].path)
    n_bins = 100
    bins = np.linspace(-2, 2, n_bins + 1)
    # store the histogram counts for a different motion parameter
    H = np.zeros((6, n_bins))
    xlist = np.empty((6, 0))
    for rp in rps:
        X = np.loadtxt(rp).T
        X[3:] *= (180. / np.pi)
        # add the histogram counts to the corresponding rows
        H += np.array([np.histogram(x, bins)[0] for x in X])
        # concatenate motion parameter data from different files into a
        # single array
        xlist = np.hstack((xlist, X))

    # Process values to get convidence intervals
    xlist.sort(1)
    left, right = int(.0005 * xlist.shape[1]), int(.9995 * xlist.shape[1])
    print('.999 confindence interval')
    print(xlist[:, left])
    # [-0.67661475 -0.82066769 -1.6521591  -1.56599341 -1.06614193 -1.09133088]
    print(xlist[:, right])
    # [1.06848    0.89511545 2.5317982  1.87914269 1.19409916 0.97554424]
    left, right = int(.005 * xlist.shape[1]), int(.995 * xlist.shape[1])
    print('.99 confindence interval')
    print(xlist[:, left])
    # [-0.46837345 -0.54565559 -1.2850646  -0.95525076 -0.70048078 -0.42188997]
    print(xlist[:, right])
    # [0.65827423 0.61529233 1.4997323  1.33685458 0.77069424 0.56606078]

    # Plot the histograms
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    H = (H.T / H.sum(1)) #normalized histogram counts for each parameter
    mbins = .5 * (bins[1:] + bins[:-1]) # bin centers
    plt.figure(figsize=(6, 4))
    #plt.plot(mbins, H, linewidth=1)
    for i, color in enumerate(colors):
        plt.plot(mbins, H[:,i], linewidth=1, color=color)
    plt.fill(mbins, H, alpha=.3)
    plt.legend(['translation x', 'translation y', 'translation z',
                'rotation x', 'rotation y', 'rotation z'], fontsize=10)
    plt.xlabel('mm/degrees')
    plt.ylabel('normalized histogram')

    # Set y-axis tick formatter to display two decimal digits
    def format_y_tick(value, _):
        return f'{value:.2f}'
    plt.gca().yaxis.set_major_formatter(
        FuncFormatter(format_y_tick))

    plt.title(f"Histogram of motion parameters")
    # plot the confidence intervals
    for i, color  in enumerate(colors):
        plt.plot([xlist[i, left], xlist[i, right]],
                 [-0.001 - .003 * i, -.001 - .003 * i], linewidth=3,
                 color=color)
    plt.plot([xlist[i, left], xlist[i, right]], [-0.018, -.018], color='w')
    plt.axis('tight')
    plt.subplots_adjust(bottom=.12, left=.14)
    plt.savefig(os.path.join(cache, f"motion_across_release3.png"),
                dpi=600)
    plt.savefig(os.path.join(cache, f"motion_across_release3.pdf"),
                dpi=600)

# %%
# ######################## GENERAL INPUTS ##############################
sub_num = [1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]
TASKS = ['ClipsTrn', 'ClipsVal', 'Raiders','WedgeAnti','WedgeClock',
         'ContRing','ExpRing']
sess_names = ["clips1","clips2", "clips3", "clips4", 
              "raiders1", "raiders2"]

cache = '/storage/store3/work/aponcema/IBC_paper3/cache_two'
mem = '/storage/store3/work/aponcema/IBC_paper3/cache_two'

sub_path = [os.path.join(DERIVATIVES, 'sub-%02d' % s) for s in sub_num]
PTS = [os.path.basename(full_path) for full_path in sub_path]
# %%
# ######################## RUN ##############################
if __name__ == '__main__':
    db = data_parser(derivatives=DERIVATIVES,subject_list = PTS,
                     task_list=TASKS,conditions=CONDITIONS)
    # %%
    # Make a sub_db with the sessions for each subject
    subject_sessions = get_subject_session(sess_names)
    subs_sess = {}

    for sub, ses in subject_sessions:
        if sub not in subs_sess:
            subs_sess[sub] = [ses]
        else:
            if ses not in subs_sess[sub]:
                subs_sess[sub].append(ses)
    subs_sess = dict(sorted(subs_sess.items()))
                     
    # %%
    new_db_ = []
    for sub in subs_sess:
        for ses in subs_sess[sub]:
            subses_db = db[(db.subject == sub) & (db.session == ses)]
            new_db_.append(subses_db)
    new_db = pd.concat(new_db_, ignore_index=True)
    
    # %%
    motion_histogram(new_db)
# %%
