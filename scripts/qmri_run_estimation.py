from ibc_public.utils_relaxo import t1_pipeline
from ibc_public.utils_relaxo import t2_pipeline
from joblib import Parallel, delayed


def run_all_estimation(sub, sess, data_root_path):
    # T2 estimation in subject space
    print('Running t2-est without spat. norm. for {}'.format(sub))
    t2_pipeline(sub_name=sub, sess_num=sess, do_plot=False, keep_tmp=False,
                root_path=data_root_path)

    # T2 estimation in MNI space
    print('Running t2-est with spat. norm. for {}'.format(sub))
    t2_pipeline(do_normalise_before=True, sub_name=sub, sess_num=sess,
                do_plot=False, keep_tmp=False, root_path=data_root_path)

    # T1 estimation in subject space
    print('Running t1-est without spat. norm. for {}'.format(sub))
    t1_pipeline(sub_name=sub, sess_num=sess, do_plot=False, keep_tmp=False,
                root_path=data_root_path)

    # T1 estimation in MNI space
    print('Running t1-est with spat. norm. for {}'.format(sub))
    t1_pipeline(do_normalise_before=True, sub_name=sub, sess_num=sess,
                do_plot=False, keep_tmp=False, root_path=data_root_path)


if __name__ == "__main__":

    DATA_ROOT = '/neurospin/ibc'

    sub_sess = {'sub-01': 'ses-21', 'sub-04': 'ses-20', 'sub-05': 'ses-22',
    'sub-06': 'ses-20', 'sub-07': 'ses-20', 'sub-08': 'ses-35',
    'sub-09': 'ses-19', 'sub-11': 'ses-17', 'sub-12': 'ses-17',
    'sub-13': 'ses-20', 'sub-14': 'ses-20', 'sub-15': 'ses-18'}

    Parallel(n_jobs=1)(delayed(run_all_estimation)(sub, sess, DATA_ROOT) for sub,
                    sess in sub_sess.items())
