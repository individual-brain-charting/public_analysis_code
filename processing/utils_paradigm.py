"""
Some utils too deal with peculiar protocols or peculiar ways of handling them

Author: Bertrand Thirion, Ana Luisa Grilo Pinho, 2015
"""
import numpy as np
from pandas import DataFrame


rsvp_language = ['consonant_strings', 'word_list', 'pseudoword_list', 'jabberwocky',
                'simple_sentence', 'probe','complex_sentence']
archi_social = ['false_belief_video', 'non_speech', 'speech', 'mechanistic_audio',
           'mechanistic_video', 'false_belief_audio', 'triangle_intention',
           'triangle_random',
       ]

relevant_conditions = {
    'emotional': ['Face', 'Shape'],
    'gambling': ['Reward', 'Punishment', 'Neutral'],
    'hcp_language': ['math', 'story'],
    'motor': ['LeftHand', 'RightHand', 'LeftFoot', 'RightFoot',
              'Cue', 'Tongue'],
    'relational': ['Relational', 'Cue', 'Control'],
    'social': ['Mental', 'Response', 'Random'],
    'wm': ['2-BackBody', '0-BackBody', '2-BackFace', '0-BackFace', '2-BackTools',
            '0-BackTools', '0-BackPlace', '2-BackPlace'],
    'archi_social': archi_social,
    'language_00': rsvp_language, 
    'language_01': rsvp_language,
    'language_02': rsvp_language,
    'language_03': rsvp_language,
    'language_04': rsvp_language,
    'language_05': rsvp_language
    }


def post_process(df, paradigm_id):
    if paradigm_id in (['language_00', 'language_01', 'language_02', 'language_03',
                        'language_04', 'language_05']):
        targets = ['complex_sentence_objrel', 'complex_sentence_objclef',
                   'complex_sentence_subjrel']
        for target in targets:
            df = df.replace(target, 'complex_sentence')
        targets = ['simple_sentence_cvp', 'simple_sentence_adj', 'simple_sentence_coord']
        for target in targets:
            df = df.replace(target, 'simple_sentence')

        # df.onset *= .001
        # df.duration = 3 * np.ones(len(df.duration))
    if paradigm_id == 'hcp_motor':
        df = df.replace('right_foot_cue', 'cue')
        df = df.replace('left_foot_cue', 'cue')
        df = df.replace('right_hand_cue', 'cue')
        df = df.replace('left_hand_cue', 'cue')
        df = df.replace('tongue_cue', 'cue')
        
    if paradigm_id in relevant_conditions.keys():
        relevant_items = relevant_conditions[paradigm_id]
        condition = np.array([df.trial_type == r for r in relevant_items]).sum(0).astype(np.bool)
        df = df[condition]

    if paradigm_id == 'PreferencePaintings':
        df['modulation'] = df['score'] - df[df.trial_type == 'painting']['score'].mean()
        df = df.fillna(1)
    elif paradigm_id == 'PreferenceFaces':
        df['modulation'] = df['score'] - df[df.trial_type == 'face']['score'].mean()
        df = df.fillna(1)
    elif paradigm_id ==  'PreferenceHouses':
        df['modulation'] = df['score'] - df[df.trial_type == 'house']['score'].mean()
        df = df.fillna(1)    
    elif paradigm_id == 'PreferenceFood':
        df['modulation'] = df['score'] - df[df.trial_type == 'food']['score'].mean()
        df = df.fillna(1)    
    return df


def make_paradigm(onset_file, paradigm_id=None):
    """ Temporary fix """
    if paradigm_id in ['wedge_clock', 'wedge_anti', 'cont_ring', 'exp_ring']:
        return None
    df = DataFrame().from_csv(onset_file, index_col=None, sep='\t')
    if 'onset' not in df.keys() and 'Onsets' in df.keys():
        df['onset'] = df['Onsets']
        df.drop('Onsets', 1, inplace=True)
    if 'duration' not in df.keys() and 'Durations' in df.keys():
        df['duration'] = df['Durations']
        df.drop('Durations', 1, inplace=True)
    if 'trial_type' not in df.keys() and 'Conditions' in df.keys():
        df['trial_type'] = df['Conditions']
        df.drop('Conditions', 1, inplace=True)
    if 'onset' not in df.keys() and 'Onset' in df.keys():
        df['onset'] = df['Onset']
        df.drop('Onset', 1, inplace=True)
    if 'duration' not in df.keys() and 'Duration' in df.keys():
        df['duration'] = df['Duration']
        df.drop('Duration', 1, inplace=True)
    if 'trial_type' not in df.keys() and 'Condition' in df.keys():
        df['trial_type'] = df['Condition']
        df.drop('Condition', 1, inplace=True)
    if 'trial_type' not in df.keys() and 'name' in df.keys():
        df['trial_type'] = df['name']
        df.drop('name', 1, inplace=True)
    df = post_process(df, paradigm_id)
    df['name'] = df['trial_type']
    return df

