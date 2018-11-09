""" This modules specifies contrasts for the archi and HCP protocols

Author: Bertrand Thirion, 2014--2015
"""

import numpy as np


def make_contrasts(paradigm_id, design_matrix_columns=None):
    """ return the contrasts matching a string"""
    if paradigm_id == 'archi_standard':
        return archi_standard(design_matrix_columns)
    elif paradigm_id == 'archi_social':
        return archi_social(design_matrix_columns)
    elif paradigm_id == 'archi_spatial':
        return archi_spatial(design_matrix_columns)
    elif paradigm_id == 'archi_emotional':
        return archi_emotional(design_matrix_columns)
    elif paradigm_id == 'hcp_emotion':
        return hcp_emotion(design_matrix_columns)
    elif paradigm_id == 'hcp_gambling':
        return hcp_gambling(design_matrix_columns)
    elif paradigm_id == 'hcp_language':
        return hcp_language(design_matrix_columns)
    elif paradigm_id == 'hcp_motor':
        return hcp_motor(design_matrix_columns)
    elif paradigm_id == 'hcp_wm':
        return hcp_wm(design_matrix_columns)
    elif paradigm_id == 'hcp_relational':
        return hcp_relational(design_matrix_columns)
    elif paradigm_id == 'hcp_social':
        return hcp_social(design_matrix_columns)
    elif paradigm_id[:9] == 'language_':
        return rsvp_language(design_matrix_columns)
    elif paradigm_id == 'colour':
        return colour(design_matrix_columns)
    elif paradigm_id in ['cont_ring', 'exp_ring', 'wedge_clock',
                         'wedge_anti']:
        return retino(design_matrix_columns)
    elif paradigm_id[:10] == 'preference':
        domain = paradigm_id[11:]
        if domain[-1] == 's':
            domain = domain[: -1]
        return preferences(design_matrix_columns, domain)
    elif paradigm_id in ['IslandWE', 'IslandWE1', 'IslandWE2', 'IslandWE3']:
        return mtt_ew(design_matrix_columns)
    elif paradigm_id in ['IslandNS', 'IslandNS1', 'IslandNS2', 'IslandNS3']:
        return mtt_ns(design_matrix_columns)
    elif paradigm_id == 'pain_localizer':
        return pain_localizer(design_matrix_columns)
    elif paradigm_id == 'movie_localizer':
        return movie_localizer(design_matrix_columns)
    elif paradigm_id == 'tom_localizer':
        return tom_localizer(design_matrix_columns)
    elif paradigm_id == 'VSTM':
        return vstm(design_matrix_columns)
    elif paradigm_id == 'enum':
        return enum(design_matrix_columns)
    elif paradigm_id == 'clips_trn':
        return dict([])
    elif paradigm_id == 'self':
        return self_localizer(design_matrix_columns)
    elif paradigm_id == 'lyon_moto':
        return lyon_moto(design_matrix_columns)
    elif paradigm_id == 'lyon_mcse':
        return lyon_mcse(design_matrix_columns)
    elif paradigm_id == 'lyon_mveb':
        return lyon_mveb(design_matrix_columns)
    elif paradigm_id == 'lyon_mvis':
        return lyon_mvis(design_matrix_columns)
    elif paradigm_id == 'audio':
        return audio(design_matrix_columns)
    else:
        raise ValueError('%s Unknown paradigm' % paradigm_id)


def _elementary_contrasts(design_matrix_columns):
    """Returns a doictionary of contrasts for all columns
        of the design matrix"""
    con = {}
    n_columns = len(design_matrix_columns)
    # simple contrasts
    for i in range(n_columns):
        con[design_matrix_columns[i]] = np.eye(n_columns)[i]
    return con


def _append_effects_interest_contrast(design_matrix_columns, contrast):
    """appends a contrast for all derivatives"""
    n_columns = len(design_matrix_columns)
    # simple contrasts
    con = []
    nuisance = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz', 'constant'] +\
               ['drift_%d' % i for i in range(20)] +\
               ['conf_%d' % i for i in range(20)]
    for i in range(n_columns):
        if design_matrix_columns[i] in nuisance:
            continue
        if len(design_matrix_columns[i]) > 11:
            if design_matrix_columns[i][-11:] == '_derivative':
                continue
        con.append(np.eye(n_columns)[i])
    if con != []:
        contrast['effects_interest'] = np.array(con)
    return contrast


def _append_derivative_contrast(design_matrix_columns, contrast):
    """appends a contrast for all derivatives"""
    n_columns = len(design_matrix_columns)
    # simple contrasts
    con = []
    for i in range(n_columns):
        if len(design_matrix_columns[i]) > 11:
            if design_matrix_columns[i][-11:] == '_derivative':
                con.append(np.eye(n_columns)[i])

    if con != []:
        contrast['derivatives'] = np.array(con)
    return contrast


def audio(design_matrix_columns):
    """Contrasts for the audio protocol"""
    contrast_names = [
        'animal-others', 'music-others', 'nature-others',
        'speech-others', 'tool-others', 'voice-others', 'mean-silence'
        ]
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    others = (con['animal'] + con['music'] + con['nature'] +
              con['speech'] + con['tool'] + con['voice']) * 1. / 5
    contrasts = {
        'mean-silence': others - con['silence'],
        'animal-others': con['animal'] - others,
        'music-others': con['music'] - others,
        'nature-others': con['nature'] - others,
        'speech-others': con['speech'] - others,
        'tool-others': con['tool'] - others,
        'voice-others': con['voice'] - others}
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def lyon_mveb(design_matrix_columns):
    """ Contrasts for Lyon motor localizer"""
    contrast_names = [
        'response', '2_letters_different-same',
        '4_letters_different-same', '6_letters_different-same',
        '6_letters_different-2_letters_different']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    #
    contrasts = {'response': con['response']}
    contrasts['2_letters_different-same'] = con['2_letters_different'] -\
        con['2_letters_same']
    contrasts['4_letters_different-same'] = con['4_letters_different'] -\
        con['4_letters_same']
    contrasts['6_letters_different-same'] = con['6_letters_different'] -\
        con['6_letters_same']
    contrasts['6_letters_different-2_letters_different'] =\
        con['6_letters_different'] - con['2_letters_different']
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def lyon_mvis(design_matrix_columns):
    """ Contrasts for Lyon motor localizer"""
    contrast_names = ['response',
                      '2_dots-2_dots_control', '4_dots-4_dots_control',
                      '6_dots-6_dots_control', '6_dots-2_dots' ]
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    # contrasts = dict([(cname, con[cname]) for cname in contrast_names[:-4]])
    contrasts = {'response': con['response']}
    contrasts['2_dots-2_dots_control'] = con['2_dots'] - con['2_dots_control']
    contrasts['4_dots-4_dots_control'] = con['4_dots'] - con['4_dots_control']
    contrasts['6_dots-6_dots_control'] = con['6_dots'] - con['6_dots_control']
    contrasts['6_dots-2_dots'] = con['6_dots'] - con['2_dots']
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def lyon_moto(design_matrix_columns):
    """ Contrasts for Lyon motor localizer"""
    contrast_names = ['instructions', 'finger_right-avg', 'finger_left-avg',
                      'foot_left-avg', 'foot_right-avg', 'hand_left-avg',
                      'hand_right-avg', 'saccade-avg', 'tongue-avg',
                      'fixation-avg', ]
    elementary_contrasts = [
        'foot_left', 'foot_right', 'finger_right', 'finger_left',
        'saccade_left', 'saccade_right', 'hand_left', 'hand_right',
        'fixation_right', 'tongue_right', 'fixation_left',  'tongue_left']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    avg = np.mean([con[cname] for cname in elementary_contrasts], 0)
    contrasts = {'instructions': con['instructions']}
    contrasts['finger_right-avg'] = con['finger_right'] - avg
    contrasts['finger_left-avg'] = con['finger_left'] - avg
    contrasts['foot_left-avg'] = con['foot_left'] - avg
    contrasts['foot_right-avg'] = con['foot_right'] - avg
    contrasts['hand_left-avg'] = con['hand_left'] - avg
    contrasts['hand_right-avg'] = con['hand_right'] - avg
    contrasts['saccade-avg'] = con['saccade_left'] + con['saccade_right']\
        - 2 * avg
    contrasts['tongue-avg'] = con['tongue_left'] + con['tongue_right']\
        - 2 * avg
    contrasts['fixation-avg'] = con['fixation_left'] + con['fixation_right']\
        - 2 * avg
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def lyon_mcse(design_matrix_columns):
    """ Contrasts for Lyon MCSE localizer"""
    contrast_names = [
        'high_salience_left', 'high_salience_right',
        'low_salience_left', 'low_salience_right', 'high-low_salience',
        'low-high_salience', 'salience_left-right', 'salience_right-left']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = {
        'high_salience_left': con['hi_salience_left'],
        'high_salience_right': con['hi_salience_right'],
        'low_salience_left': con['low_salience_left'],
        'low_salience_right': con['low_salience_right'],
        'high-low_salience':
            con['hi_salience_left'] + con['hi_salience_right'] -
            con['low_salience_left'] - con['low_salience_right'],
        'low-high_salience':
            - con['hi_salience_left'] - con['hi_salience_right']
            + con['low_salience_left'] + con['low_salience_right'],
        'salience_left-right':
            con['hi_salience_left'] - con['hi_salience_right']
            + con['low_salience_left'] - con['low_salience_right'],
        'salience_right-left':
            - con['hi_salience_left'] + con['hi_salience_right']
            - con['low_salience_left'] + con['low_salience_right'],
    }
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def self_localizer(design_matrix_columns):
    """ Contrasts for self experiment"""
    contrast_names = [
        'instructions', 'new_fa', 'old_other_hit', 'old_self_hit',
        'other_relevance_with_response', 'self_relevance_with_response',
        'self-other_hit', 'self-other_with_response',
    ]
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = {
        'instructions': con['instructions'],
        'new_fa': con['new_fa'],
        'old_other_hit': con['old_other_hit'],
        'old_self_hit': con['old_self_hit'],
        'other_relevance_with_response': con['other_relevance_with_response'],
        'self_relevance_with_response': con['self_relevance_with_response'],
        'self-other_hit': con['old_self_hit'] - con['old_other_hit'],
        'self-other_with_response':
            con['self_relevance_with_response']
            - con['other_relevance_with_response'],
    }
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def colour(design_matrix_columns):
    """ Contrasts for pain lcoalizer """
    if design_matrix_columns is None:
        return {'color': [], 'grey': [], 'color-grey': []}
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = {'color': con['color'],
                 'grey': con['grey'],
                 'color-grey': con['color'] - con['grey'],
                 }
    return contrasts


def vstm(design_matrix_columns):
    """ contrasts for vstm task, Knops protocol"""
    contrast_names = [
       'vstm_response_linear',
       'vstm_response_constant',
       'vstm_response_quadratic']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    constant = np.ones(6)
    linear = np.linspace(-1, 1, 6)
    quadratic = linear ** 2 - (linear ** 2).mean()
    con = _elementary_contrasts(design_matrix_columns)
    response = np.array([con['response_num_%d' % i]
                        for i in range(1, 7)])
    contrasts = {
        'vstm_response_constant': np.dot(constant, response),
        'vstm_response_linear': np.dot(linear, response),
        'vstm_response_quadratic': np.dot(quadratic, response),
    }
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def enum(design_matrix_columns):
    """ contrasts for vstm task, Knops protocol"""
    contrast_names = [
        'enum_response_linear',
        'enum_response_constant',
        'enum_response_quadratic']

    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    constant = np.ones(8)
    linear = np.linspace(-1, 1, 8)
    quadratic = linear ** 2 - (linear ** 2).mean()
    con = _elementary_contrasts(design_matrix_columns)
    response = np.array([con['response_num_%d' % i]
                         for i in range(1, 9)])
    contrasts = {
        'enum_response_constant': np.dot(constant, response),
        'enum_response_linear': np.dot(linear, response),
        'enum_response_quadratic': np.dot(quadratic, response),
    }
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def movie_localizer(design_matrix_columns):
    """ Contrast for pain task, TOM protocol"""
    contrast_names = ['mov_pain', 'mov_mental', 'mov_mental-pain',
                      'mov_pain-mental']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = {'mov_pain': con['pain'],
                 'mov_mental': con['mental'],
                 'mov_mental-pain': con['mental'] - con['pain'],
                 'mov_pain-mental': con['pain'] - con['mental'],
                 }
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def pain_localizer(design_matrix_columns):
    """ Contrast for pain task, TOM protocol"""
    contrast_names = ['physical_pain', 'emotional_pain',
                      'emotional-physical_pain', 'physical-emotional_pain']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = {'emotional_pain': con['emotional_pain'],
                 'physical_pain': con['physical_pain'],
                 'emotional-physical_pain':
                 con['emotional_pain'] - con['physical_pain'],
                 'physical-emotional_pain':
                 con['physical_pain'] - con['emotional_pain'],
                 }
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def tom_localizer(design_matrix_columns):
    """ Contrast for tom task, TOM protocol"""
    contrast_names = ['belief', 'photo', 'belief-photo', 'photo-belief']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = {'photo': con['photo'],
                 'belief': con['belief'],
                 'photo-belief': con['photo'] - con['belief'],
                 'belief-photo': con['belief'] - con['photo'],
                 }
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def preferences(design_matrix_columns, domain):
    """Contrast for preference experiment"""
    if domain not in ['painting', 'house', 'face', 'food']:
        raise ValueError('Not a correct domain')
    contrast_names = ['%s_linear' % domain, '%s_constant' % domain,
                      '%s_quadratic' % domain]
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = dict([(key, con[key]) for key in contrast_names])
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def _beta_contrasts(design_matrix_columns):
    """ Same as elementary contrasts, but retains only contrasts of interest"""
    con = _elementary_contrasts(design_matrix_columns)
    bad_names = tuple(['constant', 'rx', 'ry', 'rz', 'tx', 'ty', 'tz'] +
                      ['drift_%d' % d for d in range(20)] +
                      ['conf_%d' % d for d in range(20)])
    con_ = dict([(cname, cvalue) for (cname, cvalue) in con.items()
                if not cname.startswith(bad_names)])
    return con_


def mtt_ew(design_matrix_columns):
    """ Contrast for MTT north-south experiment"""
    if design_matrix_columns is None:
        return {'time_reference': [],
                'west_east_reference': [],
                'past-future_reference': [],
                'east-west_reference': [],
                'time_event': [],
                'west_east_event': [],
                'far-close_space_event': [],
                'far-close_time_event': [],
                'past-future_event': [],
                'east-west_events': [],
                'response': [],
                'cue_space': [],
                'cue_time': [],
                'average_event': [],
                'average_reference': [],
                }
    con = _beta_contrasts(design_matrix_columns)
    future_events = con['ewe_center_future_space_close'] +\
        con['ewe_center_future_space_far'] +\
        con['ewe_center_future_time_close'] +\
        con['ewe_center_future_time_far']
    past_events = con['ewe_center_past_space_close'] +\
        con['ewe_center_past_space_far'] +\
        con['ewe_center_past_time_close'] +\
        con['ewe_center_past_time_far']
    present_events =\
        con['ewe_center_present_space_close'] +\
        con['ewe_center_present_space_far'] +\
        con['ewe_center_present_time_close'] +\
        con['ewe_center_present_time_far']
    east_events =\
        con['ewe_east_present_space_close'] +\
        con['ewe_east_present_space_far'] +\
        con['ewe_east_present_time_close'] +\
        con['ewe_east_present_time_far']
    west_events =\
        con['ewe_west_present_space_close'] +\
        con['ewe_west_present_space_far'] +\
        con['ewe_west_present_time_close'] +\
        con['ewe_west_present_time_far']
    contrasts = {
        # reference
        'time_reference':
            con['rwe_center_future'] + con['rwe_center_past']
            - 2 * con['rwe_center_present'],
        'west_east_reference':
            con['rwe_east_present'] + con['rwe_west_present']
            - 2 * con['rwe_center_present'],
        'past-future_reference':
            con['rwe_center_past'] - con['rwe_center_future'],
        'east-west_reference':
            con['rwe_east_present'] - con['rwe_west_present'],
        # events
        'time_event': future_events + past_events -
            2. * (present_events + east_events + west_events) / 3,
        'west_east_event':  east_events + west_events -
            2 * present_events,  # fixme, futre, past
        'far-close_space_event':
            con['ewe_east_present_space_far']
            + con['ewe_west_present_space_far']
            - con['ewe_east_present_space_close']
            - con['ewe_west_present_space_close'],
        'far-close_time_event':
            con['ewe_center_future_time_far']
            + con['ewe_center_past_time_far']
            - con['ewe_center_future_time_close']
            - con['ewe_center_present_time_close'],
        'past-future_event': past_events - future_events,
        'east-west_events': east_events - west_events,
        'reference': np.vstack((con['rwe_center_future'],
                                con['rwe_center_past'],
                                con['rwe_center_present'],
                                con['rwe_east_present'],
                                con['rwe_west_present'])),
        'response': con['response'],
        'events': np.vstack((
            con['ewe_center_future_space_close'],
            con['ewe_center_future_space_far'],
            con['ewe_center_future_time_close'],
            con['ewe_center_future_time_far'],
            con['ewe_center_past_space_close'],
            con['ewe_center_past_space_far'],
            con['ewe_center_past_time_close'],
            con['ewe_center_past_time_far'],
            con['ewe_center_present_space_close'],
            con['ewe_center_present_space_far'],
            con['ewe_center_present_time_close'],
            con['ewe_center_present_time_far'],
            con['ewe_east_present_space_close'],
            con['ewe_east_present_space_far'],
            con['ewe_east_present_time_close'],
            con['ewe_east_present_time_far'],
            con['ewe_west_present_space_close'],
            con['ewe_west_present_space_far'],
            con['ewe_west_present_time_close'],
            con['ewe_west_present_time_far'])),
        'cue': np.vstack((con['cwe_space_close'],
                          con['cwe_space_far'],
                          con['cwe_time_close'],
                          con['cwe_time_far'])),
        'cue_space': con['cwe_space_close'] + con['cwe_space_far'],
        'cue_time': con['cwe_time_close'] + con['cwe_time_far']
        }
    contrasts['average_event'] = contrasts['events'].sum(0)
    contrasts['average_reference'] = contrasts['reference'].sum(0)
    return contrasts


def mtt_ns(design_matrix_columns):
    """ Contrast for MTT north-south experiment"""
    if design_matrix_columns is None:
        return {'time_reference': [],
                'south_north_reference': [],
                'past-future_reference': [],
                'north-south_reference': [],
                'time_event': [],
                'south_north_event': [],
                'far-close_space_event': [],
                'far-close_time_event': [],
                'past-future_event': [],
                'north-south_events': [],
                'response': [],
                'cue_space': [],
                'cue_time': [],
                'average_event': [],
                'average_reference': [],
        }
    con = _beta_contrasts(design_matrix_columns)

    future_events =\
        con['esn_center_future_space_close'] +\
        con['esn_center_future_space_far'] +\
        con['esn_center_future_time_close'] +\
        con['esn_center_future_time_far']
    past_events =\
        con['esn_center_past_space_close'] +\
        con['esn_center_past_space_far'] +\
        con['esn_center_past_time_close'] +\
        con['esn_center_past_time_far']
    present_events =\
        con['esn_center_present_space_close'] +\
        con['esn_center_present_space_far'] +\
        con['esn_center_present_time_close'] +\
        con['esn_center_present_time_far']
    north_events =\
        con['esn_north_present_space_close'] +\
        con['esn_north_present_space_far'] +\
        con['esn_north_present_time_close'] +\
        con['esn_north_present_time_far']
    south_events =\
        con['esn_south_present_space_close'] +\
        con['esn_south_present_space_far'] +\
        con['esn_south_present_time_close'] +\
        con['esn_south_present_time_far']
    contrasts = {
        # reference
        'reference': np.vstack((con['rsn_center_future'],
                                con['rsn_center_past'],
                                con['rsn_center_present'],
                                con['rsn_south_present'],
                                con['rsn_north_present'])),
        'time_reference':
            con['rsn_center_future'] + con['rsn_center_past']
            - 2 * con['rsn_center_present'],
        'south_north_reference':
            con['rsn_north_present'] + con['rsn_south_present']
            - 2 * con['rsn_center_present'],
        'past-future_reference':
            con['rsn_center_past'] - con['rsn_center_future'],
        'north-south_reference':
            con['rsn_north_present'] - con['rsn_south_present'],
        # events
        'time_event':
            future_events + past_events - 2. * (present_events),
            # + north_events + south_events) / 3,
        'south_north_event':  north_events + south_events - 2 * present_events,
        'far-close_space_event':
            con['esn_north_present_space_far']
            + con['esn_south_present_space_far']
            - con['esn_north_present_space_close']
            - con['esn_south_present_space_close'],
        'far-close_time_event':
            con['esn_center_future_time_far']
            + con['esn_center_past_time_far']
            - con['esn_center_future_space_close']
            - con['esn_center_present_space_close'],
        'past-future_event': past_events - future_events,
        'north-south_events': north_events - south_events,
        'response': con['response'],
        'events': np.vstack((
            con['esn_center_future_space_close'],
            con['esn_center_future_space_far'],
            con['esn_center_future_time_close'],
            con['esn_center_future_time_far'],
            con['esn_center_past_space_close'],
            con['esn_center_past_space_far'],
            con['esn_center_past_time_close'],
            con['esn_center_past_time_far'],
            con['esn_center_present_space_close'],
            con['esn_center_present_space_far'],
            con['esn_center_present_time_close'],
            con['esn_center_present_time_far'],
            con['esn_south_present_space_close'],
            con['esn_south_present_space_far'],
            con['esn_south_present_time_close'],
            con['esn_south_present_time_far'],
            con['esn_north_present_space_close'],
            con['esn_north_present_space_far'],
            con['esn_north_present_time_close'],
            con['esn_north_present_time_far'])),
        'cue': np.vstack((con['csn_space_close'],
                          con['csn_space_far'],
                          con['csn_time_close'],
                          con['csn_time_far'])),
        'cue_space': con['csn_space_close'] + con['csn_space_far'],
        'cue_time': con['csn_time_close'] + con['csn_time_far'],
        }
    contrasts['average_event'] = contrasts['events'].sum(0)
    contrasts['average_reference'] = contrasts['reference'].sum(0)
    return contrasts


def retino(design_matrix_columns):
    """ Contrast for retino experiment """
    if design_matrix_columns is None:
        return {'cos': [], 'sin': [], 'effects_interest': []}
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = {'cos': con['cos'],
                 'sin': con['sin'],
                 'effects_interest': np.vstack((con['cos'], con['sin'])),
                 }
    return contrasts


def rsvp_language(design_matrix_columns):
    """ Contrasts for RSVP language localizer"""
    contrast_names = [
        'complex', 'simple', 'jabberwocky', 'word_list',
        'pseudoword_list', 'consonant_string', 'complex-simple',
        'sentence-jabberwocky', 'sentence-word',
        'word-consonant_string', 'jabberworcky-pseudo',
        'word-pseudo', 'pseudo-consonant_string',
        'sentence-consonant_string', 'simple-consonant_string',
        'complex-consonant_string', 'sentence-pseudo', 'probe']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])

    con = _elementary_contrasts(design_matrix_columns)
    con['complex'] = con['complex_sentence']
    con['simple'] = con['simple_sentence']
    contrasts = {
        'complex': con['complex'],
        'simple': con['simple'],
        'probe': con['probe'],
        'jabberwocky': con['jabberwocky'],
        'word_list': con['word_list'],
        'pseudoword_list': con['pseudoword_list'],
        'consonant_string': con['consonant_strings'],
        'complex-simple': con['complex'] - con['simple'],
        'sentence-jabberwocky': (con['complex'] + con['simple']
                                 - 2 * con['jabberwocky']),
        'sentence-word': (con['complex'] + con['simple'] -
                          2 * con['word_list']),
        'word-consonant_string': con['word_list'] - con['consonant_strings'],
        'jabberworcky-pseudo': con['jabberwocky'] - con['pseudoword_list'],
        'word-pseudo': con['word_list'] - con['pseudoword_list'],
        'pseudo-consonant_string':
            con['pseudoword_list'] - con['consonant_strings'],
        'sentence-consonant_string': (con['complex'] + con['simple']
                                      - 2 * con['consonant_strings']),
        'simple-consonant_string': con['simple'] - con['consonant_strings'],
        'complex-consonant_string': con['complex'] - con['consonant_strings'],
        'sentence-pseudo':
            con['complex'] + con['simple'] - 2 * con['pseudoword_list']
    }
    assert((np.sort(contrasts.keys()) == np.sort(contrast_names)).all())
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def archi_social(design_matrix_columns):
    contrast_names = [
        'triangle_mental-random', 'false_belief-mechanistic_audio',
        'mechanistic_audio', 'false_belief-mechanistic_video',
        'mechanistic_video', 'false_belief-mechanistic',
        'speech-non_speech', 'triangle_mental', 'triangle_random',
        'false_belief_audio', 'false_belief_video',
        'speech_sound', 'non_speech_sound']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])

    con = _elementary_contrasts(design_matrix_columns)

    # and more complex / interesting ones
    contrasts = {
        'triangle_mental': con['triangle_intention'],
        'triangle_random': con['triangle_random'],
        'false_belief_audio': con['false_belief_audio'],
        'mechanistic_audio': con['mechanistic_audio'],
        'false_belief_video': con['false_belief_video'],
        'mechanistic_video': con['mechanistic_video'],
        'speech_sound': con['speech'],
        'non_speech_sound': con['non_speech'],
        'triangle_mental-random':
            con['triangle_intention'] - con['triangle_random'],
        'false_belief-mechanistic_audio':
            con['false_belief_audio'] - con['mechanistic_audio'],
        'false_belief-mechanistic_video':
            con['false_belief_video'] - con['mechanistic_video'],
        'speech-non_speech': con['speech'] - con['non_speech'],
        'mechanistic_video': con['mechanistic_video'],
        'mechanistic_audio': con['mechanistic_audio'], }
    contrasts['false_belief-mechanistic'] =\
        contrasts['false_belief-mechanistic_audio'] +\
        contrasts['false_belief-mechanistic_video']

    assert((np.sort(contrasts.keys()) == np.sort(contrast_names)).all())
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def archi_spatial(design_matrix_columns):
    contrast_names = [
        'saccades', 'rotation_hand', 'rotation_side', 'object_grasp',
        'object_orientation', 'hand-side', 'grasp-orientation']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])

    contrasts = _elementary_contrasts(design_matrix_columns)

    # more interesting contrasts
    contrasts = {
        'saccades': contrasts['saccade'],
        'rotation_hand': contrasts['rotation_hand'],
        'rotation_side': contrasts['rotation_side'],
        'object_grasp': contrasts['object_grasp'],
        'object_orientation': contrasts['object_orientation'],
        'hand-side': contrasts['rotation_hand'] - contrasts['rotation_side'],
        'grasp-orientation': (contrasts['object_grasp'] -
                              contrasts['object_orientation'])}
    assert((np.sort(contrasts.keys()) == np.sort(contrast_names)).all())
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def archi_standard(design_matrix_columns, new=True):
    contrast_names = [
        'audio_left_button_press', 'audio_right_button_press',
        'video_left_button_press', 'video_right_button_press',
        'left-right_button_press', 'right-left_button_press',
        'listening-reading', 'reading-listening',
        'motor-cognitive', 'cognitive-motor', 'reading-checkerboard',
        'horizontal-vertical', 'vertical-horizontal',
        'horizontal_checkerboard', 'vertical_checkerboard',
        'audio_sentence', 'video_sentence',
        'audio_computation', 'video_computation',
        'sentences', 'computation',
        'computation-sentences', 'sentences-computation',
    ]
    if design_matrix_columns is None:
        return dict([(x, []) for x in contrast_names])

    contrasts = _elementary_contrasts(design_matrix_columns)

    # and more complex/ interesting ones
    contrasts['audio_left_button_press'] = contrasts['audio_left_hand']
    contrasts['audio_right_button_press'] = contrasts['audio_right_hand']
    contrasts['video_left_button_press'] = contrasts['video_left_hand']
    contrasts['video_right_button_press'] = contrasts['video_right_hand']
    contrasts['audio'] = (contrasts['audio_left_hand'] +
                          contrasts['audio_right_hand'] +
                          contrasts['audio_computation'] +
                          contrasts['audio_sentence'])
    contrasts['audio_sentence'] = contrasts['audio_sentence']
    contrasts['video_sentence'] = contrasts['video_sentence']
    contrasts['audio_computation'] = contrasts['audio_computation']
    contrasts['video_computation'] = contrasts['video_computation']
    contrasts['video'] =\
        contrasts['video_left_hand'] + contrasts['video_right_hand'] + \
        contrasts['video_computation'] + contrasts['video_sentence']
    contrasts['left_button_press'] = (
        contrasts['audio_left_hand'] + contrasts['video_left_hand'])
    contrasts['right_button_press'] = (
        contrasts['audio_right_hand'] + contrasts['video_right_hand'])
    contrasts['computation'] =\
        contrasts['audio_computation'] + contrasts['video_computation']
    contrasts['sentences'] =\
        contrasts['audio_sentence'] + contrasts['video_sentence']
    contrasts['horizontal-vertical'] =\
        contrasts['horizontal_checkerboard']\
        - contrasts['vertical_checkerboard']
    contrasts['vertical-horizontal'] =\
        contrasts['vertical_checkerboard']\
        - contrasts['horizontal_checkerboard']
    contrasts['left-right_button_press'] = (
        contrasts['left_button_press'] - contrasts['right_button_press'])
    contrasts['right-left_button_press'] = (
        contrasts['right_button_press'] - contrasts['left_button_press'])
    contrasts['motor-cognitive'] = (
        contrasts['left_button_press'] + contrasts['right_button_press'] -
        contrasts['computation'] - contrasts['sentences'])
    contrasts['cognitive-motor'] = -(
        contrasts['left_button_press'] + contrasts['right_button_press'] -
        contrasts['computation'] - contrasts['sentences'])
    contrasts['listening-reading'] =\
        contrasts['audio'] - contrasts['video']
    contrasts['reading-listening'] =\
        contrasts['video'] - contrasts['audio']
    contrasts['computation-sentences'] = contrasts['computation'] -  \
        contrasts['sentences']
    contrasts['sentences-computation'] = contrasts['sentences'] -\
        contrasts['computation']
    contrasts['reading-checkerboard'] = contrasts['video_sentence'] - \
        contrasts['horizontal_checkerboard']

    contrasts = dict([(x, contrasts[x]) for x in contrast_names])
    assert((np.sort(contrasts.keys()) == np.sort(contrast_names)).all())
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def archi_emotional(design_matrix_columns):
    contrast_names = [
        'face_gender', 'face_control', 'face_trusty',
        'expression_intention', 'expression_gender', 'expression_control',
        'trusty_and_intention-control', 'trusty_and_intention-gender',
        'expression_gender-control', 'expression_intention-control',
        'expression_intention-gender', 'face_trusty-control',
        'face_gender-control', 'face_trusty-gender']
    if design_matrix_columns is None:
        return dict([(x, []) for x in contrast_names])

    contrasts = _elementary_contrasts(design_matrix_columns)

    # and more complex/ interesting ones
    contrasts = {
        'face_gender': contrasts['face_gender'],
        'face_control': contrasts['face_control'],
        'face_trusty': contrasts['face_trusty'],
        'expression_intention': contrasts['expression_intention'],
        'expression_gender': contrasts['expression_gender'],
        'expression_control': contrasts['expression_control'],
        'face_trusty-gender': (
            contrasts['face_trusty'] - contrasts['face_gender']),
        'face_gender-control': (
            contrasts['face_gender'] - contrasts['face_control']),
        'face_trusty-control': (
            contrasts['face_trusty'] - contrasts['face_control']),
        'expression_intention-gender': (
            contrasts['expression_intention'] -
            contrasts['expression_gender']),
        'expression_intention-control': (
            contrasts['expression_intention'] -
            contrasts['expression_control']),
        'expression_gender-control': (
            contrasts['expression_gender'] - contrasts['expression_control'])}
    contrasts['trusty_and_intention-gender'] = (
        contrasts['face_trusty-gender'] +
        contrasts['expression_intention-gender'])
    contrasts['trusty_and_intention-control'] = (
        contrasts['face_trusty-control'] +
        contrasts['expression_intention-control'])
    assert((np.sort(contrasts.keys()) == np.sort(contrast_names)).all())
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def hcp_emotion(design_matrix_columns=None):
    contrast_names = ['face', 'shape', 'face-shape', 'shape-face']
    if design_matrix_columns is None:
        return dict([(x, []) for x in contrast_names])
    n_columns = len(design_matrix_columns)
    contrasts = {}
    for i in range(n_columns):
        contrasts['%s' % design_matrix_columns[i].lower()] =\
            np.eye(n_columns)[i]
    contrasts = dict([(key, contrasts[key.lower()])
                     for key in ['face', 'shape']])
    contrasts = {'face-shape': contrasts['face'] - contrasts['shape'],
                 'shape-face': contrasts['shape'] - contrasts['face'],
                 'face': contrasts['face'],
                 'shape': contrasts['shape'], }
    assert((np.sort(contrasts.keys()) == np.sort(contrast_names)).all())
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def hcp_gambling(design_matrix_columns):
    contrast_names = [
        'punishment-reward', 'reward-punishment', 'punishment', 'reward']
    if design_matrix_columns is None:
        return dict([(x, []) for x in contrast_names])
    n_columns = len(design_matrix_columns)
    contrasts = {}
    for i in range(n_columns):
        contrasts['%s' % design_matrix_columns[i].lower()] =\
            np.eye(n_columns)[i]
    contrasts = dict([(key, contrasts[key])
                     for key in ['punishment', 'reward']])
    contrasts = {
        'punishment-reward': contrasts['punishment'] - contrasts['reward'],
        'reward-punishment': contrasts['reward'] - contrasts['punishment'],
        'punishment': contrasts['punishment'],
        'reward': contrasts['reward']}
    assert((np.sort(contrasts.keys()) == np.sort(contrast_names)).all())
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def hcp_language(design_matrix_columns):
    contrast_names = ['math-story', 'story-math', 'math', 'story']
    if design_matrix_columns is None:
        return dict([(x, []) for x in contrast_names])
    n_columns = len(design_matrix_columns)
    contrasts = {}
    for i in range(n_columns):
        contrasts['%s' % design_matrix_columns[i].lower()] =\
            np.eye(n_columns)[i]
    contrasts = dict([(key, contrasts[key]) for key in ['math', 'story']])
    contrasts = {
        'math-story': contrasts['math'] - contrasts['story'],
        'story-math': contrasts['story'] - contrasts['math'],
        'math': contrasts['math'],
        'story': contrasts['story']}
    assert((np.sort(contrasts.keys()) == np.sort(contrast_names)).all())
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def hcp_motor(design_matrix_columns):
    contrast_names = [
        'left_hand', 'right_hand', 'left_foot', 'right_foot',
        'tongue', 'tongue-avg', 'left_hand-avg', 'right_hand-avg',
        'left_foot-avg', 'right_foot-avg', 'cue']
    if design_matrix_columns is None:
        return dict([(x, []) for x in contrast_names])
    n_columns = len(design_matrix_columns)
    contrasts = {}
    for i in range(n_columns):
        contrasts['%s' % design_matrix_columns[i].lower()] =\
            np.eye(n_columns)[i]
    contrasts['Average'] = (
        contrasts['left_hand'] + contrasts['right_hand'] +
        contrasts['left_foot'] +
        contrasts['right_foot'] + contrasts['tongue']) / 5
    contrasts = {
        'cue': contrasts['cue'],
        'left_hand': contrasts['left_hand'],
        'right_hand': contrasts['right_hand'],
        'left_foot': contrasts['left_foot'],
        'right_foot': contrasts['right_foot'],
        'tongue': contrasts['tongue'],
        'left_hand-avg': contrasts['left_hand'] - contrasts['Average'],
        'right_hand-avg': contrasts['right_hand'] - contrasts['Average'],
        'left_foot-avg': contrasts['left_foot'] - contrasts['Average'],
        'right_foot-avg': contrasts['right_foot'] - contrasts['Average'],
        'tongue-avg': contrasts['tongue'] - contrasts['Average']
    }
    assert((np.sort(contrasts.keys()) == np.sort(contrast_names)).all())
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def hcp_relational(design_matrix_columns):
    contrast_names = ['relational', 'relational-match', 'match']
    if design_matrix_columns is None:
        return dict([(x, []) for x in contrast_names])
    n_columns = len(design_matrix_columns)
    contrasts = {}
    for i in range(n_columns):
        contrasts['%s' % design_matrix_columns[i].lower()] =\
            np.eye(n_columns)[i]
    contrasts = dict([(key, contrasts[key]) for key in
                      ['relational', 'control']])
    contrasts = {
        'match': contrasts['control'],
        'relational': contrasts['relational'],
        'relational-match': contrasts['relational'] - contrasts['control']}
    assert((np.sort(contrasts.keys()) == np.sort(contrast_names)).all())
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def hcp_social(design_matrix_columns):
    contrast_names = ['mental-random', 'mental', 'random']
    if design_matrix_columns is None:
        return dict([(x, []) for x in contrast_names])
    n_columns = len(design_matrix_columns)
    contrasts = {}
    for i in range(n_columns):
        contrasts['%s' % design_matrix_columns[i].lower()] =\
            np.eye(n_columns)[i]
    contrasts = dict([(key, contrasts[key]) for key in ['mental', 'random']])
    contrasts = {
        'mental-random': contrasts['mental'] - contrasts['random'],
        'random': contrasts['random'],
        'mental': contrasts['mental'], }
    assert((np.sort(contrasts.keys()) == np.sort(contrast_names)).all())
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def hcp_wm(design_matrix_columns):
    contrast_names = ['2back-0back', '0back-2back', 'body-avg',
                      'face-avg', 'place-avg', 'tools-avg',
                      '0back_body', '2back_body', '0back_face', '2back_face',
                      '0back_tools', '2back_tools', '0back_place',
                      '2back_place']
    if design_matrix_columns is None:
        return dict([(x, []) for x in contrast_names])
    n_columns = len(design_matrix_columns)
    contrasts = {}
    for i in range(n_columns):
        contrasts['%s' % design_matrix_columns[i].lower()] =\
            np.eye(n_columns)[i]
    contrasts = dict([(key, contrasts[key]) for key in [
        '2back_body', '0back_body', '2back_face', '0back_face', '2back_tools',
        '0back_tools', '0back_place', '2back_place']])
    contrasts['2back'] = (contrasts['2back_body'] + contrasts['2back_face'] +
                          contrasts['2back_tools'] + contrasts['2back_place'])
    contrasts['0back'] = (contrasts['0back_body'] + contrasts['0back_face'] +
                          contrasts['0back_tools'] + contrasts['0back_place'])
    contrasts['body'] = (contrasts['2back_body'] + contrasts['0back_body']) / 2
    contrasts['face'] = (contrasts['2back_face'] + contrasts['0back_face']) / 2
    contrasts['place'] = (
        contrasts['2back_place'] + contrasts['0back_place']) / 2
    contrasts['tools'] = (
        contrasts['2back_tools'] + contrasts['0back_tools']) / 2
    contrasts['average'] = (contrasts['2back'] + contrasts['0back']) / 8
    contrasts = {
        '0back_body': contrasts['0back_body'],
        '2back_body': contrasts['2back_body'],
        '0back_face': contrasts['0back_face'],
        '2back_face': contrasts['2back_face'],
        '0back_tools': contrasts['0back_tools'],
        '2back_tools': contrasts['2back_tools'],
        '0back_place': contrasts['0back_place'],
        '2back_place': contrasts['2back_place'],
        '2back-0back': contrasts['2back'] - contrasts['0back'],
        '0back-2back': contrasts['0back'] - contrasts['2back'],
        'body-avg': contrasts['body'] - contrasts['average'],
        'face-avg': contrasts['face'] - contrasts['average'],
        'place-avg': contrasts['place'] - contrasts['average'],
        'tools-avg': contrasts['tools'] - contrasts['average']}
    assert((np.sort(contrasts.keys()) == np.sort(contrast_names)).all())
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts
