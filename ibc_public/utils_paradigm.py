"""
Some utils too deal with peculiar protocols or peculiar ways of handling them

Author: Bertrand Thirion, Ana Luisa Grilo Pinho, 2015
"""
import numpy as np
from pandas import read_csv, concat


rsvp_language = ['consonant_strings', 'word_list', 'pseudoword_list',
                 'jabberwocky', 'simple_sentence', 'probe', 'complex_sentence']
archi_social = [
    'false_belief_video', 'non_speech', 'speech', 'mechanistic_audio',
    'mechanistic_video', 'false_belief_audio', 'triangle_intention',
    'triangle_random', ]

relevant_conditions = {
    'emotional': ['Face', 'Shape'],
    'gambling': ['Reward', 'Punishment', 'Neutral'],
    'hcp_language': ['math', 'story'],
    'motor': ['LeftHand', 'RightHand', 'LeftFoot', 'RightFoot',
              'Cue', 'Tongue'],
    'relational': ['Relational', 'Cue', 'Control'],
    'social': ['Mental', 'Response', 'Random'],
    'wm': ['2-BackBody', '0-BackBody', '2-BackFace', '0-BackFace',
           '2-BackTools', '0-BackTools', '0-BackPlace', '2-BackPlace'],
    'archi_social': archi_social,
    'language_00': rsvp_language,
    'language_01': rsvp_language,
    'language_02': rsvp_language,
    'language_03': rsvp_language,
    'language_04': rsvp_language,
    'language_05': rsvp_language,
    'language': rsvp_language,
    }


def post_process(df, paradigm_id):
    language_paradigms = ['language_%02d' % i for i in range(6)] +\
                         ['rsvp-language', 'language_', 'language']
    if paradigm_id in language_paradigms:
        targets = ['complex_sentence_objrel',
                   'complex_sentence_objclef',
                   'complex_sentence_subjrel']
        for target in targets:
            df = df.replace(target, 'complex_sentence')
        targets = ['simple_sentence_cvp',
                   'simple_sentence_adj',
                   'simple_sentence_coord']
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

    if paradigm_id == 'lyon-visu':
        df = df.replace('visage', 'face')

    if paradigm_id in relevant_conditions.keys():
        relevant_items = relevant_conditions[paradigm_id]
        condition = np.array(
            [df.trial_type == r for r in relevant_items])\
            .sum(0).astype(np.bool)
        df = df[condition]

    if paradigm_id[:10] == 'preference':
        domain = paradigm_id[11:]
        if domain[-1] == 's':
            domain = domain[:-1]
        #
        linear = df[df.trial_type == domain]['score']
        mean = linear.mean()
        linear -= mean
        df['modulation'] = linear
        df = df.fillna(1)
        # add a regressor with constant values
        df2 = df[df.trial_type == domain]
        df2.modulation = np.ones_like(linear)
        df2.trial_type = '%s_constant' % domain
        # add quadratic regressor
        df3 = df[df.trial_type == domain]
        quadratic = linear ** 2
        quadratic -= quadratic.mean()
        quadratic -= (linear * np.dot(quadratic, linear) /
                      np.dot(linear, linear))
        df3.modulation = quadratic
        df3.trial_type = '%s_quadratic' % domain
        df = df.replace(domain, '%s_linear' % domain)
        df = concat([df, df2, df3], axis=0, ignore_index=True)
    responses_we = ['response_we_east_present_space_close',
                    'response_we_west_present_space_far',
                    'response_we_center_past_space_far',
                    'response_we_west_present_time_close',
                    'response_we_east_present_time_far',
                    'response_we_center_past_space_close',
                    'response_we_center_present_space_close',
                    'response_we_center_present_space_far',
                    'response_we_center_present_time_far',
                    'response_we_east_present_time_close',
                    'response_we_center_past_time_close',
                    'response_we_center_past_time_far',
                    'response_we_east_present_space_far',
                    'response_we_center_future_time_far',
                    'response_we_center_future_time_far',
                    'response_we_center_future_time_close',
                    'response_we_west_present_space_close',
                    'response_we_center_present_time_close',
                    'response_we_center_present_time_close',
                    'response_we_center_future_space_far',
                    'response_we_center_future_space_close',
                    'response_we_west_present_time_far']

    if paradigm_id == 'IslandWE':
        for response in responses_we:
            df = df.replace(response, 'response')

    responses_sn = ['response_sn_north_present_space_far',
                    'response_sn_south_present_time_close',
                    'response_sn_center_present_space_close',
                    'response_sn_south_present_time_far',
                    'response_sn_center_future_space_close',
                    'response_sn_center_past_space_close',
                    'response_sn_north_present_time_close',
                    'response_sn_center_past_space_far',
                    'response_sn_south_present_space_close',
                    'response_sn_center_present_time_far',
                    'response_sn_center_past_time_far',
                    'response_sn_center_future_space_far',
                    'response_sn_center_future_space_far',
                    'response_sn_center_future_time_close',
                    'response_sn_center_past_time_close',
                    'response_sn_north_present_time_far',
                    'response_sn_south_present_space_far',
                    'response_sn_center_present_time_close',
                    'response_sn_north_present_space_close',
                    'response_sn_center_present_space_far',
                    'response_sn_center_future_time_far',
                    'response_sn_center_future_time_far']

    if paradigm_id == 'IslandNS':
        for response in responses_sn:
            df = df.replace(response, 'response')

    if paradigm_id == 'enum':
        for i in range(1, 9):
            df = df.replace('memorization_num_%d' % i, 'response_num_%d' % i)
    if paradigm_id == 'VSTM':
        for i in range(1, 7):
            df = df.replace('memorization_num_%d' % i, 'response_num_%d' % i)

    instructions = ['Ins_bouche', 'Ins_index', 'Ins_jambe',
                    'Ins_main', 'Ins_repos', 'Ins_yeux', ]
    if paradigm_id == 'lyon-moto':
        for instruction in instructions:
            df = df.replace(instruction, 'instructions')
        df = df.replace('sacaade_right', 'saccade_right')
        df = df.replace('sacaade_left', 'saccade_left')
        # df = df.replace('Bfix', 'fixation')
        df = df[df.trial_type != 'Bfix']

    if paradigm_id == 'lyon-mcse':
        df = df[df.trial_type != 'Bfix']

    if paradigm_id == 'lyon-lec1':
        df = df[df.trial_type != 'Bfix']
        df = df[df.trial_type != 'start_random_string']
        df = df[df.trial_type != 'start_pseudoword']
        df = df[df.trial_type != 'start_word']

    if paradigm_id == 'lyon-lec2':
        df = df[df.trial_type != 'Bfix']
        df = df[df.trial_type != 'Suite']

    if paradigm_id == 'lyon-visu':
        df = df[df.trial_type != 'Bfix']

    if paradigm_id == 'lyon-audi':
        df = df[df.trial_type != 'Bfix']
        df = df[df.trial_type != 'start_sound']
        df = df[df.trial_type != 'cut']
        df = df[df.trial_type != '1']

    if paradigm_id == 'lyon-mvis':
        df = df[df.trial_type != 'grid']
        df = df[df.trial_type != 'Bfix']
        df = df[df.trial_type != 'maintenance']

    if paradigm_id == 'lyon-mveb':
        df = df[df.trial_type != 'cross']
        df = df[df.trial_type != 'blank2']

    if paradigm_id == 'audio':
        voices = ['voice_%d' % i for i in range(60)]
        musics = ['music_%d' % i for i in range(60)]
        animals = ['animal_%d' % i for i in range(60)]
        speeches = ['speech_%d' % i for i in range(60)]
        natures = ['nature_%d' % i for i in range(60)]
        tools = ['tools_%d' % i for i in range(60)]
        for voice in voices:
            df = df.replace(voice, 'voice')
        for animal in animals:
            df = df.replace(animal, 'animal')
        for music in musics:
            df = df.replace(music, 'music')
        for speech in speeches:
            df = df.replace(speech, 'speech')
        for nature in natures:
            df = df.replace(nature, 'nature')
        for tool in tools:
            df = df.replace(tool, 'tool')
    if paradigm_id == 'attention':
        df = df[df.trial_type.isin([
            'spatial_incongruent', 'double_congruent', 'spatial_congruent',
            'double_incongruent', 'spatialcue', 'doublecue'])]
    if paradigm_id == 'stop-signal':
        df = df[df.trial_type.isin(['go', 'stop'])]
    if paradigm_id in ['ward-aliport', 'ward-and-aliport']:
        df = df[df.trial_type.isin([
            'PA_with_intermediate', 'PA_without_intermediate',
            'UA_with_intermeidate', 'UA_without_intermeidate'
        ])]
        df.replace('PA_with_intermediate', 'ambiguous_intermediate',
                   inplace=True)
        df.replace('PA_without_intermediate', 'ambiguous_direct',
                   inplace=True)
        df.replace('UA_with_intermeidate', 'unambiguous_intermediate',
                   inplace=True)
        df.replace('UA_without_intermeidate', 'unambiguous_direct',
                   inplace=True)
    if paradigm_id == 'two-by-two':
        df = df[df.trial_type.isin([
            'taskstay_cuestay', 'taskswitch_cueswitch', 'taskswitch_cuestay',
            'taskstay_cueswitch'])]
    if paradigm_id == 'discount':
        df = df[df.trial_type.isin(['stim'])]
        df1 = df.copy()
        df1['modulation'] = df1['large_amount']
        df1.drop('later_delay', 1, inplace=True)
        df1.drop('large_amount', 1, inplace=True)
        df1.replace('stim', 'amount', inplace=True)
        df2 = df.copy()
        df2['modulation'] = df2['later_delay']
        df2.drop('large_amount', 1, inplace=True)
        df2.drop('later_delay', 1, inplace=True)
        df2.replace('stim', 'delay', inplace=True)
        df = concat([df1, df2], axis=0, ignore_index=True)
    if paradigm_id == 'selective-stop-signal':
        df = df[df.trial_type.isin(['go', 'ignore', 'stop'])]
    if paradigm_id == 'stroop':
        df = df[df.trial_type.isin(['congruent', 'incongruent'])]
    if paradigm_id == 'columbia-cards':
        df = df[df.trial_type.isin(['card_flip'])]
        df1 = df.copy()
        df1.replace('card_flip', 'gain', inplace=True)
        df1['modulation'] = df1['gain_amount']
        df2 = df.copy()
        df2.replace('card_flip', 'loss', inplace=True)
        df2['modulation'] = df2['loss_amount']
        df3 = df.copy()
        df3.replace('card_flip', 'num_loss_cards', inplace=True)
        df3['modulation'] = df3['num_loss_cards']
        df = concat([df1, df2, df3], axis=0, ignore_index=True)
        df.drop('loss_amount', 1, inplace=True)
        df.drop('num_loss_cards', 1, inplace=True)
        df.drop('gain_amount', 1, inplace=True)
    if paradigm_id == 'dot-patterns':
        df.replace('cue_AX', 'cue', inplace=True)
        df.replace('cue_BX', 'cue', inplace=True)
        df.replace('cue_AY', 'cue', inplace=True)
        df.replace('cue_BY', 'cue', inplace=True)
        df = df[df.trial_type.isin([
            'probe_BY', 'probe_AY', 'probe_BX', 'probe_AX', 'cue'])]
        df.replace('probe_AX', 'correct_cue_correct_probe', inplace=True)
        df.replace('probe_BX', 'incorrect_cue_correct_probe', inplace=True)
        df.replace('probe_AY', 'correct_cue_incorrect_probe', inplace=True)
        df.replace('probe_BY', 'incorrect_cue_incorrect_probe', inplace=True)
    return df


def make_paradigm(onset_file, paradigm_id=None):
    """ Temporary fix """
    if paradigm_id in ['wedge_clock', 'wedge_anti', 'cont_ring', 'exp_ring']:
        return None
    df = read_csv(onset_file, index_col=None, sep='\t')
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
