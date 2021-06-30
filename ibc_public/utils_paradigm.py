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

    if paradigm_id == 'lyon_visu':
        df = df.replace('visage', 'face')
    if paradigm_id == 'lyon_audi':
        df = df.replace('envir', 'environment')
    if paradigm_id == '':
        pass

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

    ###
    if paradigm_id == 'MTTNS':
        for response in responses_sn:
            df = df.replace(response, 'response')

    if paradigm_id == 'MTTWE':
        for response in responses_we:
            df = df.replace(response, 'response')
    ###

    if paradigm_id == 'enumeration':
        for i in range(1, 9):
            df = df.replace('memorization_num_%d' % i, 'response_num_%d' % i)
    if paradigm_id == 'VSTM':
        for i in range(1, 7):
            df = df.replace('memorization_num_%d' % i, 'response_num_%d' % i)

    if paradigm_id == 'self':
        df = df.replace('self_relevance_with_response', 'encode_self')
        df = df.replace('other_relevance_with_response', 'encode_other')
        df = df.replace('self_relevance_with_no_response',
                        'encode_self_no_response')
        df = df.replace('other_relevance_with_no_response',
                        'encode_other_no_reponse')
        df = df.replace('old_self_hit', 'recognition_self_hit')
        df = df.replace('old_self_miss', 'recognition_self_miss')
        df = df.replace('old_other_hit', 'recognition_other_hit')
        df = df.replace('old_other_miss', 'recognition_other_miss')
        df = df.replace('new_fa', 'false_alarm')
        df = df.replace('new_cr', 'correct_rejection')
        df = df.replace('old_self_no_response', 'recognition_self_no_response')
        df = df.replace('old_other_no_response',
            'recognition_other_no_response')

    instructions = ['Ins_bouche', 'Ins_index', 'Ins_jambe',
                    'Ins_main', 'Ins_repos', 'Ins_yeux', ]
    if paradigm_id == 'lyon_moto':
        for instruction in instructions:
            df = df.replace(instruction, 'instructions')
        df = df.replace('sacaade_right', 'saccade_right')
        df = df.replace('sacaade_left', 'saccade_left')
        # df = df.replace('Bfix', 'fixation')
        df = df[df.trial_type != 'Bfix']

    if paradigm_id == 'lyon_mcse':
        df = df[df.trial_type != 'Bfix']

    if paradigm_id == 'lyon_lec1':
        df = df[df.trial_type != 'Bfix']
        df = df[df.trial_type != 'start_random_string']
        df = df[df.trial_type != 'start_pseudoword']
        df = df[df.trial_type != 'start_word']

    if paradigm_id == 'lyon_lec2':
        df = df[df.trial_type != 'Bfix']
        df = df[df.trial_type != 'Suite']

    if paradigm_id == 'lyon_visu':
        df = df[df.trial_type != 'Bfix']

    if paradigm_id == 'lyon_audi':
        df = df[df.trial_type != 'Bfix']
        df = df[df.trial_type != 'start_sound']
        df = df[df.trial_type != 'cut']
        df = df[df.trial_type != '1']

    if paradigm_id == 'lyon_mvis':
        df = df[df.trial_type != 'grid']
        df = df[df.trial_type != 'Bfix']
        df = df[df.trial_type != 'maintenance']

    if paradigm_id == 'lyon_mveb':
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
    if paradigm_id == 'stop_signal':
        df = df[df.trial_type.isin(['go', 'stop'])]
    if paradigm_id in ['ward-aliport', 'ward_and_aliport']:
        df = df[df.trial_type.isin([
            'planning_PA_with_intermediate',
            'planning_PA_without_intermediate',
            'planning_UA_with_intermediate',
            'planning_UA_without_intermediate',
            'move_PA_with_intermediate',
            'move_PA_without_intermediate',
            'move_UA_with_intermediate',
            'move_UA_without_intermediate'])]

        df.replace('planning_PA_with_intermediate',
                   'planning_ambiguous_intermediate', inplace=True)
        df.replace('planning_PA_without_intermediate',
                   'planning_ambiguous_direct', inplace=True)
        df.replace('planning_UA_with_intermediate',
                   'planning_unambiguous_intermediate', inplace=True)
        df.replace('planning_UA_without_intermediate',
                   'planning_unambiguous_direct', inplace=True)
        df.replace('move_PA_with_intermediate',
                   'move_ambiguous_intermediate', inplace=True)
        df.replace('move_PA_without_intermediate',
                   'move_ambiguous_direct', inplace=True)
        df.replace('move_UA_with_intermediate',
                   'move_unambiguous_intermediate', inplace=True)
        df.replace('move_UA_without_intermediate', 'move_unambiguous_direct',
                   inplace=True)

    if paradigm_id == 'two_by_two':
        df = df[df.trial_type.isin([
            'cue_taskstay_cuestay',
            'cue_taskstay_cueswitch',
            'cue_taskswitch_cuestay',
            'cue_taskswitch_cueswitch',
            'stim_taskstay_cuestay',
            'stim_taskstay_cueswitch',
            'stim_taskswitch_cuestay',
            'stim_taskswitch_cueswitch'])]
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
    if paradigm_id == 'selective_stop_signal':
        df = df[df.trial_type.isin(['go_critical', 'go_noncritical',
                                    'ignore_noncritical',
                                    'stop_critical'])]
        df = df.replace('ignore_noncritical', 'ignore')
        df = df.replace('stop_critical', 'stop')
    if paradigm_id == 'stroop':
        df = df[df.trial_type.isin(['congruent', 'incongruent'])]
    if paradigm_id == 'columbia_cards':
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
    if paradigm_id == 'dot_patterns':
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

    if paradigm_id == 'biological_motion1':
        df = df[df.trial_type.isin(['global_upright', 'global_inverted',
                                    'natural_upright', 'natural_inverted'])]
    if paradigm_id == 'biological_motion2':
        df = df[df.trial_type.isin(['modified_upright', 'modified_inverted',
                                    'natural_upright', 'natural_inverted'])]
    if paradigm_id == 'mathlang':
        trial_types = [
            'colorlessg_auditory', 'colorlessg_visual',
            'wordlist_auditory', 'wordlist_visual',
            'arithmetic_fact_auditory', 'arithmetic_fact_visual',
            'arithmetic_principle_auditory', 'arithmetic_principle_visual',
            'theory_of_mind_auditory', 'theory_of_mind_visual',
            'geometry_fact_visual', 'geometry_fact_auditory',
            'general_visual', 'general_auditory',
            'context_visual', 'context_auditory']
        df = df[df.trial_type.isin(trial_types)]
    if paradigm_id == 'spatial_navigation':
        for intersection_ in ['intersection_%d' % i for i in range(4)]:
            df.replace(intersection_, 'intersection', inplace=True)
        trial_types = ['encoding_phase', 'navigation', 'experimental',
                       'pointing_experimental', 'control', 'pointing_control',
                       'intersection']
        df = df[df.trial_type.isin(trial_types)]
    if paradigm_id == 'EmoMem':
        pass
    if paradigm_id == 'EmoReco':
        pass
    if paradigm_id == 'StopNogo':
        pass
    if paradigm_id == 'Catell':
        pass
    if paradigm_id == 'RewProc':
        df.drop(df[df.trial_type == 'prefix'].index, 0, inplace=True)
        df.drop(df[df.trial_type == 'postfix'].index, 0, inplace=True)
        green = [tt for tt in df.trial_type.unique() if 'green' in tt]
        left = [tt for tt in df.trial_type.unique() if 'left' in tt]
        stay = [tt for tt in df.trial_type.unique() if 'stay' in tt]
        switch = [tt for tt in df.trial_type.unique() if 'switch' in tt]
        resp = [tt for tt in df.trial_type.unique() if 'resp' in tt]
        df1 = df.copy()
        df1 = df1[df.trial_type.isin(green)]
        df1.trial_type = 'green'
        df2 = df.copy()
        df2 = df2[df.trial_type.isin(left)]
        df2.trial_type = 'left'
        df3 = df.copy()
        df3 = df3[df.trial_type.isin(switch)]
        df3.trial_type = 'switch'
        df4 = df.copy()
        df4 = df4[df.trial_type.isin(stay)]
        df4.trial_type = 'stay'
        df.drop(df[df.trial_type.isin(resp)].index, 0, inplace=True)
        df = concat([df, df1, df2, df3, df4], axis=0, ignore_index=True)
    if paradigm_id == 'NARPS':
        df.drop(df[df.trial_type == 'fix'].index, 0, inplace=True)
        stim = [tt for tt in df.trial_type.unique() if 'stim' in tt]
        resp = [tt for tt in df.trial_type.unique() if 'stim' not in tt]
        df1 = df.copy()
        df1 = df1[df.trial_type.isin(stim)]
        df2 = df1.copy()
        df1['modulation'] = [float(x.split('+')[1].split('_')[0])
                             for x in df1.trial_type.values]
        df2['modulation'] = [float(x.split('-')[1])
                             for x in df2.trial_type.values]
        df1.trial_type = 'gain'
        df2.trial_type = 'loss'
        df = df[df.trial_type.isin(resp)]
        df['modulation'] = 1
        df = concat([df, df1, df2], axis=0, ignore_index=True)
    return df


def make_paradigm(onset_file, paradigm_id=None):
    """ Temporary fix """
    # if paradigm_id in ['wedge_clock', 'wedge_anti', 'cont_ring', 'exp_ring']:
    #     return None
    df = read_csv(onset_file, index_col=None, sep='\t', na_values=['NaN'],
                  keep_default_na=False)
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
