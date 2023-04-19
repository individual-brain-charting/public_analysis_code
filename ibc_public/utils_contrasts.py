""" This modules specifies contrasts for the IBC tasks

Author: Bertrand Thirion, Ana Luisa Pinho 2014--2020
"""

import numpy as np


def make_contrasts(paradigm_id, design_matrix_columns=None):
    """ return the contrasts matching a string"""
    if paradigm_id == 'ArchiStandard':
        return archi_standard(design_matrix_columns)
    elif paradigm_id == 'ArchiSocial':
        return archi_social(design_matrix_columns)
    elif paradigm_id == 'ArchiSpatial':
        return archi_spatial(design_matrix_columns)
    elif paradigm_id == 'ArchiEmotional':
        return archi_emotional(design_matrix_columns)
    elif paradigm_id == 'HcpEmotion':
        return hcp_emotion(design_matrix_columns)
    elif paradigm_id == 'HcpGambling':
        return hcp_gambling(design_matrix_columns)
    elif paradigm_id == 'HcpLanguage':
        return hcp_language(design_matrix_columns)
    elif paradigm_id == 'HcpMotor':
        return hcp_motor(design_matrix_columns)
    elif paradigm_id == 'HcpWm':
        return hcp_wm(design_matrix_columns)
    elif paradigm_id == 'HcpRelational':
        return hcp_relational(design_matrix_columns)
    elif paradigm_id == 'HcpSocial':
        return hcp_social(design_matrix_columns)
    elif paradigm_id == 'RSVPLanguage':
        return rsvp_language(design_matrix_columns)
    elif paradigm_id in ['ContRing', 'ExpRing', 'WedgeClock', ###
                         'WedgeAnti', 'Wedge', 'Ring']:###
        return retino(design_matrix_columns)###
    elif paradigm_id in ['Wedge', 'WedgeAnti', 'WedgeClock']:
        return wedge(design_matrix_columns)
    elif paradigm_id in ['Ring', 'ContRing', 'ExpRing']:
        return ring(design_matrix_columns)
    elif paradigm_id[:10] == 'Preference':
        domain = paradigm_id[10:].lower()
        if domain[-1] == 's':
            domain = domain[: -1]
        return preferences(design_matrix_columns, domain)
    elif paradigm_id == 'MTTWE':
        return mtt_we_relative(design_matrix_columns)
    elif paradigm_id == 'MTTNS':
        return mtt_sn_relative(design_matrix_columns)
    elif paradigm_id == 'EmotionalPain':
        return emotional_pain(design_matrix_columns)
    elif paradigm_id == 'PainMovie':
        return pain_movie(design_matrix_columns)
    elif paradigm_id == 'TheoryOfMind':
        return theory_of_mind(design_matrix_columns)
    elif paradigm_id == 'VSTM':
        return vstm(design_matrix_columns)
    elif paradigm_id == 'Enumeration':
        return enumeration(design_matrix_columns)
    elif paradigm_id == 'ClipsTrn':
        return dict([])
    elif paradigm_id == 'Self':
        return self_localizer(design_matrix_columns)
    elif paradigm_id == 'Moto':
        return lyon_moto(design_matrix_columns)
    elif paradigm_id == 'MCSE':
        return lyon_mcse(design_matrix_columns)
    elif paradigm_id == 'MVEB':
        return lyon_mveb(design_matrix_columns)
    elif paradigm_id == 'MVIS':
        return lyon_mvis(design_matrix_columns)
    elif paradigm_id == 'Lec1':
        return lyon_lec1(design_matrix_columns)
    elif paradigm_id == 'Lec2':
        return lyon_lec2(design_matrix_columns)
    elif paradigm_id == 'Audi':
        return lyon_audi(design_matrix_columns)
    elif paradigm_id == 'Visu':
        return lyon_visu(design_matrix_columns)
    elif paradigm_id == 'Audio':
        return audio(design_matrix_columns)
    elif paradigm_id == 'Bang':
        return bang(design_matrix_columns)
    elif paradigm_id == 'SelectiveStopSignal':
        return selective_stop_signal(design_matrix_columns)
    elif paradigm_id == 'StopSignal':
        return stop_signal(design_matrix_columns)
    elif paradigm_id == 'Stroop':
        return stroop(design_matrix_columns)
    elif paradigm_id == 'Discount':
        return discount(design_matrix_columns)
    elif paradigm_id == 'Attention':
        return attention(design_matrix_columns)
    elif paradigm_id == 'WardAndAllport':
        return towertask(design_matrix_columns)
    elif paradigm_id == 'TwoByTwo':
        return two_by_two(design_matrix_columns)
    elif paradigm_id == 'ColumbiaCards':
        return columbia_cards(design_matrix_columns)
    elif paradigm_id == 'DotPatterns':
        return dot_patterns(design_matrix_columns)
    elif paradigm_id == 'BiologicalMotion1':
        return biological_motion1(design_matrix_columns)
    elif paradigm_id == 'BiologicalMotion2':
        return biological_motion2(design_matrix_columns)
    elif paradigm_id == 'MathLanguage':
        return math_language(design_matrix_columns)
    elif paradigm_id == 'SpatialNavigation':
        return spatial_navigation(design_matrix_columns)
    elif paradigm_id == 'EmoMem':
        return emotional_memory(design_matrix_columns)
    elif paradigm_id == 'EmoReco':
        return emotion_recognition(design_matrix_columns)
    elif paradigm_id == 'StopNogo':
        return stop_nogo(design_matrix_columns)
    elif paradigm_id == 'Catell':
        return oddball(design_matrix_columns)
    elif paradigm_id == 'VSTMC':
        return vstmc(design_matrix_columns)
    elif paradigm_id == 'FingerTapping':
        return finger_tapping(design_matrix_columns)
    elif paradigm_id == 'RewProc':
        return reward_processing(design_matrix_columns)
    elif paradigm_id == 'NARPS':
        return narps(design_matrix_columns)
    elif paradigm_id == 'FaceBody':
        return face_body(design_matrix_columns)
    elif paradigm_id == 'Scene':
        return scenes(design_matrix_columns)
    elif paradigm_id == 'BreathHolding':
        return breath_holding(design_matrix_columns)
    elif paradigm_id == 'Checkerboard':
        return checkerboard(design_matrix_columns)
    elif paradigm_id == 'FingerTap':
        return fingertap(design_matrix_columns)
    elif paradigm_id == 'ItemRecognition':
        return item_recognition(design_matrix_columns)
    elif paradigm_id == 'VisualSearch':
        return search(design_matrix_columns)
    elif paradigm_id == 'Color':
        return color(design_matrix_columns)
    elif paradigm_id == 'Motion':
        return motion(design_matrix_columns)
    elif paradigm_id == 'OptimismBias':
        return optimism_bias(design_matrix_columns)
    elif paradigm_id == 'HarririAomic':
        return harriri_aomic(design_matrix_columns)
    elif paradigm_id == 'FacesAomic':
        return faces_aomic(design_matrix_columns)
    elif paradigm_id == 'StroopAomic':
        return stroop_aomic(design_matrix_columns)
    elif paradigm_id == 'WorkingMemoryAomic':
        return working_memory_aomic(design_matrix_columns)    
    elif paradigm_id == 'Emotion':
        return emotion(design_matrix_columns)
    elif paradigm_id == 'MDTB':
        return mdtb(design_matrix_columns)
    elif paradigm_id == 'MultiModal':
        return multi_modal(design_matrix_columns)
    elif paradigm_id == 'Abstraction':
        return abstraction(design_matrix_columns)
    elif paradigm_id == 'AbstractionLocalizer':
        return abstraction_localizer(design_matrix_columns)
    elif paradigm_id == 'Mario':
        return mario(design_matrix_columns)
    else:
        raise ValueError('%s Unknown paradigm' % paradigm_id)


def _elementary_contrasts(design_matrix_columns):
    """Returns a dictionary of contrasts for all columns
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


def mario(design_matrix_columns):
    """ Contrasts for Mario task """
    contrast_names = [
        # 'action_fire',
        'action_jump',
        'action_leftrun',
        'action_leftwalk',
        # 'action_pipedown',
        'action_rest',
        'action_rightrun',
        'action_rightwalk',
        'loss_dying',
        # 'loss_powerdown',
        # 'loss_powerup_miss',
        'onscreen_enemy',
        'onscreen_powerup',
        #'reward_bricksmash',
        'reward_coin',
        'reward_enemykill_impact',
        'reward_enemykill_kick',
        'reward_enemykill_stomp',
        'reward_powerup_taken',
        'action',
        'loss',
        'reward',
        'reward-loss',
    ]
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = dict([(name, con[name]) for name in contrast_names[:-4]])
    contrasts['action'] = np.sum([con[name] for name in con.keys()
                                  if 'action' in name], 0)
    contrasts['loss'] = np.sum([con[name] for name in con.keys()
                               if 'loss' in name], 0)
    contrasts['reward'] = np.sum([con[name] for name in con.keys()
                                 if 'reward' in name], 0)
    contrasts['reward-loss']  = contrasts['reward'] - contrasts['loss']
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def emotion(design_matrix_columns):
    """ Contrasts for emotion task """
    contrast_names = ['neutral_image', 'negative_image', 'echelle_valence',
                      'negative-neutral'
    ]
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = dict([(name, con[name]) for name in contrast_names[:3]])
    contrasts['negative-neutral'] = con['negative_image'] - con['neutral_image']
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def multi_modal(design_matrix_columns):
    """ Contrasts for multimodal task (Leuven protocol)"""
    audio_ = ['audio_animal', 'audio_monkey', 'audio_nature', 'audio_silence',
             'audio_speech', 'audio_tools', 'audio_voice']
    tactile_ = ['tactile_bottom', 'tactile_middle', 'tactile_top']
    image_ = ['image_animals', 'image_birds', 'image_fruits',
              'image_human_body', 'image_human_face', 'image_human_object',
              'image_monkey_body','image_monkey_face', 'image_monkey_object',
              'image_sculpture']
    others_ = ['audio', 'audio-control', 'visual', 'visual-control',
               'tactile', 'tactile-control', 'audio-visual', 'visual-audio',
               'tactile-visual', 'visual-tactile', 'tactile-audio',
               'audio-tactile', 'face-other', 'body-other', 'body-non_face',
               'animate-inanimate', 'monkey_speech-other', 'speech-other',
               'speech+voice-other'
    ]
    contrast_names = audio_ + tactile_ + image_ + others_
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = dict([(name, con[name]) for name in audio_ + tactile_ + image_])
    audio = np.sum([con[x] for x in audio_], 0)
    visual = np.sum([con[x] for x in image_], 0)
    tactile = np.sum([con[x] for x in tactile_], 0)
    face_ = ['image_human_face', 'image_monkey_face']
    body_ = ['image_human_body', 'image_monkey_body']
    animate_ = face_ + body_ + ['image_animals', 'image_birds']
    face = np.sum([con[x] for x in face_], 0)
    body = np.sum([con[x] for x in body_], 0)
    animate = np.sum([con[x] for x in animate_], 0)
    contrasts['audio'] = audio - con['audio_silence']
    contrasts['audio-control'] = audio - 7 * con['audio_silence']
    contrasts['visual'] = visual
    contrasts['visual-control'] = visual - 10 * con['audio_silence']
    contrasts['tactile'] = tactile
    contrasts['tactile-control'] = tactile - 3 * con['silence']
    contrasts['audio-visual'] = audio - visual
    contrasts['visual-audio'] = visual - audio
    contrasts['tactile-visual'] = tactile - visual
    contrasts['visual-tactile'] = visual - tactile
    contrasts['tactile-audio'] = tactile - audio
    contrasts['audio-tactile'] = audio - tactile
    contrasts['face-other'] = 5 * face - visual
    contrasts['body-other'] =  5 * body - visual
    contrasts['body-non_face'] = 4 * body + face - visual
    contrasts['animate-inanimate'] = 5 * animate - 3 * visual
    contrasts['monkey_speech-other'] = 7 * con['audio_monkey'] - audio
    contrasts['speech-other'] = 7 * con['audio_speech'] - audio
    contrasts['speech+voice-other'] =\
        3 * (con['audio_speech'] + con['audio_voice']) - audio

    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def abstraction_localizer(design_matrix_columns):
    """ Contrasts for Abstraction Localizer"""
    localizer_ = ['localizer_faces', 'localizer_humanbody',
                      'localizer_words', 'localizer_nonsensewords',
                      'localizer_numbers', 'localizer_places',
                      'localizer_objects', 'localizer_checkerboards']
    others_ = ['response', 'localizer_faces-other',
               'localizer_humanbody-other','localizer_words-other',
               'localizer_nonsensewords-other',
               'localizer_numbers-other',
               'localizer_places-other', 'localizer_objects-other',
               'localizer_checkerboards-other'
    ]
    contrast_names = localizer_ + others_
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = dict([(name, con[name]) for name in localizer_])
    localizer = np.sum([con[x] for x in localizer_], 0)
    contrasts['localizer_faces-other'] = 8 * con['localizer_faces'] -\
        localizer
    contrasts['localizer_humanbody-other'] = 8 *\
        con['localizer_humanbody'] - localizer
    contrasts['localizer_words-other'] = 8 * con['localizer_words'] -\
        localizer
    contrasts['localizer_nonsensewords-other'] = 8 *\
        con['localizer_nonsensewords'] - localizer
    contrasts['localizer_numbers-other'] = 8 * con['localizer_numbers'] -\
        localizer
    contrasts['localizer_places-other'] = 8 * con['localizer_places'] -\
        localizer
    contrasts['localizer_objects-other'] = 8 * con['localizer_objects'] -\
        localizer
    contrasts['localizer_checkerboards-other'] = 8 *\
        con['localizer_checkerboards'] - localizer
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def abstraction(design_matrix_columns):
    """ Contrasts for Abstraction """
    humanbody_ = ['humanbody_geometry', 'humanbody_edge',
                      'humanbody_photo']
    animals_ = ['animals_geometry', 'animals_edge', 'animals_photo']
    faces_ = ['faces_geometry', 'faces_edge', 'faces_photo']
    flora_ =['flora_geometry', 'flora_edge', 'flora_photo']
    objects_ = ['objects_geometry', 'objects_edge', 'objects_photo']
    places_ = ['places_geometry', 'places_edge', 'places_photo']
    others_ = ['humanbody-other', 'animals-other','faces-other',
               'flora-other','objects-other','places-other',
               'geometry-other','edge-other','photo-other',
               'humanbody_geometry-humanbody_other',
               'humanbody_edge-humanbody_other',
               'humanbody_photo-humanbody_other',
               'animals_geometry-animals_other',
               'animals_edge-animals_other',
               'animals_photo-animals_other',
               'faces_geometry-faces_other',
               'faces_edge-faces_other',
               'faces_photo-faces_other',
               'flora_geometry-flora_other',
               'flora_edge-flora_other',
               'flora_photo-flora_other',
               'objects_geometry-objects_other',
               'objects_edge-objects_other',
               'objects_photo-objects_other',
               'places_geometry-places_other',
               'places_edge-places_other',
               'places_photo-places_other',
               'response'
    ]
    contrast_names = humanbody_ + animals_ + faces_ + flora_ +\
                    objects_ + places_ + others_
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = dict([(name, con[name]) for name in humanbody_ +\
                      animals_ + faces_ + flora_ +\
                      objects_ + places_])
    humanbody = np.sum([con[x] for x in humanbody_], 0)
    animals = np.sum([con[x] for x in animals_], 0)
    faces = np.sum([con[x] for x in faces_], 0)
    flora = np.sum([con[x] for x in flora_], 0)
    objects = np.sum([con[x] for x in objects_], 0)
    places = np.sum([con[x] for x in places_], 0)
    allstim_ =  faces_ + humanbody_ + animals_ + flora_ + objects_  +\
        places_
    geometry_ = ['humanbody_geometry', 'animals_geometry', 'faces_geometry',
                 'flora_geometry','objects_geometry', 'places_geometry']
    edge_ = ['humanbody_edge', 'animals_edge', 'faces_edge',
                 'flora_edge','objects_edge', 'places_edge']
    photo_ = ['humanbody_photo', 'animals_photo', 'faces_photo',
                 'flora_photo','objects_photo', 'places_photo']
    allstim = np.sum([con[x] for x in allstim_], 0)
    geometry = np.sum([con[x] for x in geometry_], 0)
    edge = np.sum([con[x] for x in edge_], 0)
    photo = np.sum([con[x] for x in photo_], 0)
    contrasts['humanbody-other'] = 6 * humanbody - allstim
    contrasts['animals-other'] = 6 * animals - allstim
    contrasts['faces-other'] = 6 * faces- allstim
    contrasts['flora-other'] = 6 * flora - allstim
    contrasts['places-other'] = 6 * places - allstim
    contrasts['objects-other'] = 6 * objects - allstim
    contrasts['geometry-other'] = 3 * geometry - allstim
    contrasts['edge-other'] = 3 * edge - allstim
    contrasts['photo-other'] = 3 * photo - allstim
    contrasts['humanbody_geometry-humanbody_other'] = 3 *\
        con['humanbody_geometry'] - humanbody
    contrasts['humanbody_edge-humanbody_other'] = 3 *\
        con['humanbody_edge'] - humanbody
    contrasts['humanbody_photo-humanbody_other'] = 3 *\
        con['humanbody_photo'] - humanbody
    contrasts['animals_geometry-animals_other'] = 3 *\
        con['animals_geometry'] - animals
    contrasts['animals_edge-animals_other'] = 3 *\
        con['animals_edge'] - animals
    contrasts['animals_photo-animals_other'] = 3 *\
        con['animals_photo'] - animals
    contrasts['faces_geometry-faces_other'] = 3 *\
        con['faces_geometry'] - faces
    contrasts['faces_edge-faces_other'] = 3 *\
        con['faces_edge'] - faces
    contrasts['faces_photo-faces_other'] = 3 *\
        con['faces_photo'] - faces
    contrasts['flora_geometry-flora_other'] = 3 *\
        con['flora_geometry'] - flora
    contrasts['flora_edge-flora_other'] = 3 *\
        con['flora_edge'] - flora
    contrasts['flora_photos-flora_other'] = 3 *\
        con['flora_photos'] - flora
    contrasts['objects_geometry-objects_other'] = 3 *\
        con['objects_geometry'] - objects
    contrasts['objects_edge-objects_other'] = 3 *\
        con['objects_edge'] - objects
    contrasts['objects_photo-objects_other'] = 3 *\
        con['objects_photo'] - objects
    contrasts['places_geometry-places_other'] = 3 *\
        con['places_geometry'] - places
    contrasts['places_edge-places_other'] = 3 *\
        con['places_edge'] - places
    contrasts['places_photo-places_other'] = 3 *\
        con['places_photo'] - places
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def mdtb(design_matrix_columns):
    """ Contrasts for Multy Domain Task Battery """
    contrast_names = ['action_action', 'action_control',
                      'finger_simple', 'finger_complex',
                      'semantic_hard', 'semantic_easy', #####
                      '2back_easy', '2back_hard', ####
                      'tom_photo', 'tom_belief', ####
                      'search_easy', 'search_hard',  ####
                      'flexion_extension',
                      'action_action-control',
                      'finger_complex-simple',
                      'semantic_hard-easy',
                      '2back_hard-easy',
                      'tom_belief-photo',
                      'search_hard-easy'
    ]
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = dict([(name, con[name]) for name in contrast_names[:13]])
    contrasts['action_action-control'] = con['action_action'] - con['action_control']
    contrasts['finger_complex-simple'] = con['finger_simple'] - con['finger_complex']
    contrasts['semantic_hard-easy'] = con['semantic_hard'] - con['semantic_easy']
    contrasts['2back_hard-easy'] = con['2back_hard'] - con['2back_easy']
    contrasts['tom_belief-photo'] = con['tom_belief'] - con['tom_photo']
    contrasts['search_hard-easy'] =  con['search_hard'] - con['search_easy']
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def working_memory_aomic(design_matrix_columns):
    """ Contrasts for WorkingMemory in AOMIC """
    contrast_names = ['active_change', 'active_no_change',
                       'passive_change', 'passive_no_change',
                       'active-passive','active_change-active_no_change'
    ]
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = dict([(name, con[name]) for name in contrast_names[:4]])
    contrasts['active-passive'] = con['active_change'] +\
        con['active_no_change'] - con['passive_change'] -\
        con['passive_no_change']
    contrasts['active_change-active_no_change'] = con['active_change'] -\
        con['active_no_change']
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def stroop_aomic(design_matrix_columns):
    """ Contrasts for StroopAomic """
    contrast_names = ['incongruent_word_male_face_female',
                      'congruent_word_female_face_female',
                      'congruent_word_male_face_male',
                      'incongruent_word_female_face_male',
                      'index_response', 'middle_response',
                      'congurent-incongruent',
                      'incongurent-congruent',
                      'face_male-face_female',
                      'word_male-word_female',
                      'index-middle', 'middle-index'
    ]
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = dict([(name, con[name]) for name in contrast_names[:6]])
    congruent = con['congruent_word_female_face_female'] +\
                con['congruent_word_male_face_male']
    incongruent = con['incongruent_word_male_face_female'] +\
                  con['incongruent_word_female_face_male']
    face_male = con['incongruent_word_female_face_male'] +\
                con['congruent_word_male_face_male']
    face_female = con['congruent_word_female_face_female'] +\
                  con['incongruent_word_male_face_female']
    word_male = con['congruent_word_male_face_male'] +\
                con['incongruent_word_male_face_female']
    word_female = con['congruent_word_female_face_female'] +\
                  con['incongruent_word_female_face_male']
    contrasts['congurent-incongruent'] = congruent - incongruent
    contrasts['incongurent-congruent'] = incongruent - congruent
    contrasts['face_male-face_female'] = face_male - face_female
    contrasts['word_male-word_female'] = word_male - word_female
    contrasts['index-middle'] = con['index_response'] - con['middle_response']
    contrasts['middle-index'] = con['middle_response'] - con['index_response']
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def harriri_aomic(design_matrix_columns):
    """ Contrasts for HarririAomic """
    contrast_names = ['emotion', 'index_response',  'middle_response',
                      'shape','emotion-shape']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = {'emotion': con['emotion'],
                 'index_response': con['index_response'],
                 'middle_response': con['middle_response'],
                 'shape' : con['shape'],
                 'emotion-shape': con['emotion'] - con['shape']}
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def faces_aomic(design_matrix_columns):
    """ Contrasts for color localizer """
    contrast_names = ['ITI',
                      'anger', 'contempt', 'joy', 'neutral', 'pride',
                      'all-neutral',
                      'anger-neutral',
                      'contempt-neutral',
                      'joy-neutral',
                      'pride-neutral',
                      'male-female',
                      'female-male',
                      'mediterranean-european',
                      'european-mediterranean']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    anger = np.sum([con[x] for x in con.keys() if x[:5] == 'anger'], 0)
    contempt = np.sum([con[x] for x in con.keys() if x[:8] == 'contempt'], 0)
    joy = np.sum([con[x] for x in con.keys() if x[:3] == 'joy'], 0)
    neutral = np.sum([con[x] for x in con.keys() if x[:7] == 'neutral'], 0)
    pride = np.sum([con[x] for x in con.keys() if x[:5] == 'pride'], 0)
    male = np.sum([con[x] for x in con.keys() if 'female' not in x], 0)
    female  = np.sum([con[x] for x in con.keys() if 'female' in x], 0)
    european = np.sum([con[x] for x in con.keys() if 'european' not in x], 0)
    mediterranean = np.sum(
        [con[x] for x in con.keys() if 'mediterranean' in x], 0)
    contrasts = {'ITI': con ['ITI'],
                 'anger': anger,
                 'contempt': contempt,
                 'joy': joy,
                 'neutral': neutral,
                 'pride': pride,
                 'anger-neutral': anger - neutral,
                 'contempt-neutral': contempt - neutral,
                 'joy-neutral': joy - neutral,
                 'pride-neutral': pride - neutral,
                 'all-neutral': .25 * (anger + contempt + joy + pride) - neutral,
                 'female-male': female-male,
                 'male-female':male-female,
                 'european-mediterranean': european - mediterranean,
                 'mediterranean-european': mediterranean - european,
    }
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def color(design_matrix_columns):
    """ Contrasts for color localizer """
    contrast_names = ['chromatic', 'achromatic', 'chromatic-achromatic',
                      'response']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = {'chromatic': con['chromatic'],
                 'achromatic': con['achromatic'],
                 'chromatic-achromatic': con['chromatic'] - con['achromatic'],
                 'response': con['y']
                 }
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def optimism_bias(design_matrix_columns):
    """ Contrasts for optimism bias protocol """
    contrast_names = ['all_events', 'optimism_bias', 'future_vs_past',
                      'positive_vs_negative', 'future_positive_vs_negative',
                      'past_positive_vs_negative', 'interaction']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    all_ = [c for c in ['past_positive', 'future_positive', 'past_negative', 'future_negative',
                        'inconclusive'] if c in design_matrix_columns]
    all_events = np.mean([con[c] for c in all_ ], 0)
    future_negative_control = [c for c in
                               ['future_positive', 'past_positive', 'past_negative']
                               if c in design_matrix_columns]
    if 'future_negative' in design_matrix_columns:
        optimism_bias = con['future_negative'] - np.mean(
            [con[c] for c in future_negative_control], 0)
    elif 'future_neutral' in design_matrix_columns:
        optimism_bias = con['future_neutral'] - np.mean(
            [con[c] for c in future_negative_control], 0)
    else:
        optimism_bias = con['future_positive'] - np.mean(
            [con[c] for c in future_negative_control], 0)
    future = [s for s in ['future_negative', 'future_positive']
              if s in design_matrix_columns] #  'future_neutral',
    past = [s for s in ['past_negative', 'past_positive']
            if s in design_matrix_columns] # 'past_neutral',
    positive = [s for s in ['future_positive', 'past_positive']
                if s in design_matrix_columns]
    negative = [s for s in ['future_negative', 'past_negative']
                if s in design_matrix_columns]
    positive_vs_negative = np.mean([con[c] for c in positive], 0) -\
                           np.mean([con[c] for c in negative], 0)
    future_vs_past = np.mean([con[c] for c in future], 0)\
                     - np.mean([con[c] for c in past], 0)
    inter_pos = [c for c in ['future_positive', 'past_negative']
                 if c in design_matrix_columns]
    inter_neg = [c for c in ['future_negative', 'past_positive']
                 if c in design_matrix_columns]
    interaction = np.mean([con[c] for c in inter_pos], 0)\
                     - np.mean([con[c] for c in inter_neg], 0)
    if 'future_negative' in design_matrix_columns:
        future_positive_vs_negative = con['future_positive'] - con['future_negative']
    elif 'future_neutral' in design_matrix_columns:
        future_positive_vs_negative = con['future_positive'] - con['future_neutral']
    elif 'past_negative' in design_matrix_columns:
        future_positive_vs_negative = con['future_positive'] - con['past_negative']
    else:
        future_positive_vs_negative = con['future_positive']
    if 'past_negative' in design_matrix_columns:
        past_positive_vs_negative = con['past_positive'] - con['past_negative']
    else:
        past_positive_vs_negative = positive_vs_negative
    contrasts = {'all_events': all_events - con['fix'],
                 'optimism_bias': optimism_bias,
                 'future_vs_past': future_vs_past,
                 'positive_vs_negative': positive_vs_negative,
                 'future_positive_vs_negative': future_positive_vs_negative,
                 'past_positive_vs_negative': past_positive_vs_negative,
                 'interaction': interaction
                 }
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def motion(design_matrix_columns):
    """ Contrasts for motion localizer """
    contrast_names = [#'left_incoherent',  'left_coherent_clock', 'left_stationary',
                      #'left_coherent_anti',
                      #'right_incoherent',  'right_coherent_clock', 'right_stationary',
                      #'right_coherent_anti',
                      #'both_incoherent',  'both_coherent_clock', 'both_stationary',
                      #'both_coherent_anti',
                      'incoherent',  'coherent', 'stationary',
                      'clock', 'anti', 'response',
                      'coherent-incoherent', 'coherent-stationary',
                      'incoherent-stationary', 'clock-anti', 'left-right']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    incoherent = con['left_incoherent'] + con['right_incoherent'] + con['both_incoherent']
    stationary = con['left_stationary'] + con['right_stationary'] + con['both_stationary']
    clock = (con['left_coherent_clock'] + con['right_coherent_clock']
             + con['both_coherent_clock'])
    if 'left_coherent_anti' in design_matrix_columns:
        anti = (con['left_coherent_anti'] + con['right_coherent_anti']
                + con['both_coherent_anti'])
        left = con['left_incoherent'] + con['left_coherent_clock'] +\
               con['left_stationary'] + con['left_coherent_anti']
    else:
        anti = 1.5 * (con['right_coherent_anti'] + con['both_coherent_anti'])
        left = con['left_incoherent'] + con['left_coherent_clock'] +\
               con['left_stationary']

    coherent = clock + anti
    right = con['right_incoherent'] + con['right_coherent_clock'] +\
            con['right_stationary'] + con['right_coherent_anti']

    #
    contrasts = {'incoherent': incoherent,
                 'coherent': coherent,
                 'clock': clock,
                 'anti': anti,
                 'stationary': stationary,
                 'response':  con['y'],
                 'coherent-incoherent': coherent - incoherent,
                 'coherent-stationary': coherent - stationary,
                 'incoherent-stationary': incoherent-stationary,
                 'clock-anti': clock - anti,
                 'left-right': left - right}
    #
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def search(design_matrix_columns):
    """ Contrasts for search protocol"""
    contrast_names = ['memory_array_four',
                      'memory_array_two',
                      'response_hit',
                      'response_miss',
                      'sample_item',
                      'delay_vis',
                      'delay_wm',
                      'probe_item',
                      'probe_item_absent',
                      'probe_item_present',
                      'probe_item_four',
                      'probe_item_two',
                      'search_array',
                      'search_array_four',
                      'search_array_two',
                      'search_array_absent',
                      'search_array_present',
                      'delay_vis-delay_wm',
                      'probe_item_absent-probe_item_present',
                      'search_array_absent-search_array_present',
                      'probe_item_four-probe_item_two',
                      'search_array_four-search_array_two'
    ]
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = {
        'probe_item': con['probe_item_two_present'] +
                      con['probe_item_four_present'] +
                      con['probe_item_two_absent'] +
                      con['probe_item_four_absent'],
        'search_array': con['search_array_two_present'] +
                        con['search_array_four_present'] +
                        con['search_array_two_absent'] +
                        con['search_array_four_absent'],
        'probe_item_absent': con['probe_item_two_absent'] +
                             con['probe_item_four_absent'],
        'probe_item_present': con['probe_item_two_present'] +
                             con['probe_item_four_present'],
        'search_array_absent': con['search_array_two_absent'] +
                               con['search_array_four_absent'],
        'search_array_present': con['search_array_two_present'] +
                               con['search_array_four_present'],
        'probe_item_four': con['probe_item_four_present'] +
                           con['probe_item_four_absent'],
        'probe_item_two': con['probe_item_two_present'] +
                          con['probe_item_two_absent'],
        'search_array_four': con['search_array_four_absent'] +
                             con['search_array_four_present'],
        'search_array_two': con['search_array_two_absent'] +
                             con['search_array_two_present'],
        'delay_vis-delay_wm': con['delay_vis'] - con['delay_wm']
        }
    for name in contrast_names[:7]:
        contrasts[name] = con[name]
    contrasts['probe_item_absent-probe_item_present'] = \
        contrasts['probe_item_absent'] - contrasts['probe_item_present']
    contrasts['search_array_absent-search_array_present'] = \
        contrasts['search_array_absent'] - contrasts['search_array_present']
    contrasts['probe_item_four-probe_item_two'] = \
        contrasts['probe_item_four'] - contrasts['probe_item_two']
    contrasts['search_array_four-search_array_two'] = \
        contrasts['search_array_four'] - contrasts['search_array_two']
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def breath_holding(design_matrix_columns):
    """ Contrasts for breath holding protocol"""
    contrast_names = ['breathe', 'hold_breath', 'hold-breathe', 'breathe-hold']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = dict([(name, con[name]) for name in contrast_names[:2]])
    contrasts['hold-breathe'] = con['hold_breath'] - con['breathe']
    contrasts['breathe-hold'] = - contrasts['hold-breathe']
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def checkerboard(design_matrix_columns):
    """ Contrasts for breath holding protocol"""
    contrast_names = ['checkerboard-fixation']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = {'checkerboard-fixation': con['checkerboard'] - con['fixation']}
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def fingertap(design_matrix_columns):
    """ Contrasts for breath holding protocol"""
    contrast_names = ['fingertap-rest']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = {'fingertap-rest': con['fingertap'] - con['rest']}
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def item_recognition(design_matrix_columns):
    """ Contrasts for breath holding protocol"""
    contrast_names = ['encode5-encode1', 'probe5_mem-probe1_mem',
                      'probe5_new-probe1_new', 'prob-arrow',
                      'encode', 'arrow_left-arrow_right']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = {
        'encode5-encode1': con['encode5'] - con['encode1'],
        'probe5_mem-probe1_mem': con['probe5_mem'] - con['probe1_mem'],
        'probe5_new-probe1_new': con['probe5_new'] - con['probe1_new'],
        'prob-arrow': con['probe1_mem'] + con['probe1_new'] + con['probe3_mem']
                      + con['probe3_new'] + con['probe5_mem']
                      + con['probe5_new'] - 3 * con['arrow_right']
                      - 3 * con['arrow_left'],
        'encode': con['encode1'] + con['encode3'] + con['encode5'],
        'arrow_left-arrow_right': con['arrow_left'] - con['arrow_right']
    }
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def narps(design_matrix_columns):
    """ Contrasts for reward processing experiment"""
    contrast_names = ['gain', 'loss', 'weakly_accept', 'weakly_reject',
                      'strongly_accept', 'strongly_reject',
                      'reject-accept', 'accept-reject']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = dict([(name, con[name]) for name in contrast_names[:6]])
    contrasts['reject-accept'] = con['weakly_reject'] + con['strongly_reject']\
        - con['weakly_accept'] - con['strongly_accept']
    contrasts['accept-reject'] = - contrasts['reject-accept']
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def scenes(design_matrix_columns):
    """Contrasts for scenes protocol"""
    contrast_names = [
        'dot_easy_left', 'dot_easy_right', 'dot_hard_left', 'dot_hard_right',
        'scene_impossible_correct', 'scene_impossible_incorrect',
        'scene_possible_correct', # 'scene_possible_incorrect',
        'scene_possible_correct-scene_impossible_correct',
        'scene_correct-dot_correct',
        'dot_left-right',
        'dot_hard-easy'
        ]
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    if 'scene_possible_incorrect' in design_matrix_columns:
        scene_correct_minus_dot_correct = (
            con['scene_impossible_correct'] + con['scene_possible_correct'] -
            con['scene_impossible_incorrect'] - con['scene_possible_incorrect'])
    else:
        scene_correct_minus_dot_correct = (
            con['scene_impossible_correct'] + con['scene_possible_correct'] -
            2 * con['scene_impossible_incorrect'])
    contrasts = dict([(name, con[name]) for name in contrast_names[:7]])
    contrasts['scene_possible_correct-scene_impossible_correct'] =\
        con['scene_possible_correct'] - con['scene_impossible_correct']
    contrasts['scene_correct-dot_correct'] = scene_correct_minus_dot_correct
    contrasts['dot_left-right'] =\
        con['dot_easy_left'] + con['dot_hard_left'] -\
        con['dot_easy_right'] - con['dot_hard_right']
    contrasts['dot_hard-easy'] =\
        -con['dot_easy_left'] + con['dot_hard_left'] -\
        con['dot_easy_right'] + con['dot_hard_right']
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def face_body(design_matrix_columns):
    """ Contrasts for FaceBody task"""
    contrast_names = [
        'bodies_body', 'bodies_limb',
        'characters_number', 'characters_word',
        'faces_adult', 'faces_child',
        'objects_car', 'objects_instrument',
        'places_corridor', 'places_house',
        'bodies-others', 'characters-others', 'faces-others',
        'objects-others', 'places-others']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = dict([(name, con[name]) for name in contrast_names[:10]])
    mean_ = np.sum(list(contrasts.values()), 0)
    bodies = con['bodies_body'] + con['bodies_limb']
    characters = con['characters_number'] + con['characters_word']
    faces = con['faces_adult'] + con['faces_child']
    objects = con['objects_car'] + con['objects_instrument']
    places = con['places_corridor'] + con['places_house']
    contrasts['bodies-others'] = 5 * bodies - mean_
    contrasts['characters-others'] = 5 * characters - mean_
    contrasts['faces-others'] = 5 * faces - mean_
    contrasts['objects-others'] = 5 * objects - mean_
    contrasts['places-others'] = 5 * places - mean_
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def reward_processing(design_matrix_columns):
    """ Contrasts for reward processing experiment"""
    contrast_names = [
        'stim', 'minus_20', 'plus_20', 'minus_10', 'plus_10',
        'green-purple', 'purple-green', 'left-right', 'right-left',
        'switch', 'stay', 'switch-stay', 'stay-switch',
        'gain', 'loss', 'gain-loss', 'loss-gain'
    ]
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    gain = con['plus_20'] + con['plus_10']
    loss = con['minus_20'] + con['minus_10']
    contrasts = dict([(name, con[name]) for name in contrast_names[:5]])
    contrasts['green-purple'] = con['green']
    contrasts['purple-green'] = - con['green']
    contrasts['left-right'] = con['left']
    contrasts['right-left'] = - con['left']
    contrasts['switch'] = con['switch']
    contrasts['stay'] = con['stay']
    contrasts['switch-stay'] = con['switch'] - con['stay']
    contrasts['stay-switch'] = con['stay'] - con['switch']
    contrasts['gain'] = gain
    contrasts['loss'] = loss
    contrasts['gain-loss'] = gain-loss
    contrasts['loss-gain'] = loss-gain
    for name in ['minus_20', 'plus_20', 'minus_10', 'plus_10']:
        contrasts[name] = con[name]
    """
    contrast_names = [
        'out_+10', 'out_+20', 'out_-10', 'out_-20', 'stim',
        'resp_green-left_switch', 'resp_green-right_init',
        'resp_green-right_stay', 'resp_green-right_switch',
        'resp_purple-left_stay', 'resp_purple-left_switch',
        'resp_purple-right_stay', 'resp_purple-right_switch',
        'gain-loss', 'loss-gain', 'stay-switch', 'switch-stay',
        # 'gain-loss_stay', 'loss-gain_stay',
        # 'gain-loss_switch', 'loss-gain_switch',
        'green-purple', 'purple-green',
        'left-right', 'right-left']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = dict([(name, con[name]) for name in contrast_names[:13]])
    stay = con['resp_green-right_stay'] + con['resp_purple-left_stay']\
        + con['resp_purple-right_stay']
    switch = con['resp_green-left_switch'] + con['resp_green-right_switch']\
        + con['resp_purple-left_switch'] + con['resp_purple-right_switch']
    contrasts['stay-switch'] = stay - switch
    contrasts['switch-stay'] = switch - stay
    contrasts['gain-loss'] = con['out_+10'] + con['out_+20']\
        - con['out_-10'] - con['out_-20']
    contrasts['loss-gain'] = - contrasts['gain-loss']
    green = con['resp_green-left_switch'] + con['resp_green-right_init']\
        + con['resp_green-right_stay'] + con['resp_green-right_switch']
    purple = con['resp_purple-left_stay'] + con['resp_purple-left_switch']\
        + con['resp_purple-right_stay'] + con['resp_purple-right_switch']
    left = con['resp_green-left_switch'] + con['resp_purple-left_stay']\
        + con['resp_purple-left_switch']
    right = con['resp_green-right_stay'] + con['resp_green-right_switch']\
        + con['resp_purple-right_stay'] + con['resp_purple-right_switch']
    contrasts['green-purple'] = green - purple
    contrasts['purple-green'] = - contrasts['green-purple']
    contrasts['left-right'] = left - right
    contrasts['right-left'] = - contrasts['left-right']
    """
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def math_language(design_matrix_columns):
    """ Contrasts for math-language task"""
    contrast_names = [
        'colorlessg_auditory', 'colorlessg_visual',
        'wordlist_auditory', 'wordlist_visual',
        'arithmetic_fact_auditory', 'arithmetic_fact_visual',
        'arithmetic_principle_auditory', 'arithmetic_principle_visual',
        'theory_of_mind_auditory', 'theory_of_mind_visual',
        'geometry_fact_auditory', 'geometry_fact_visual',
        'general_auditory', 'general_visual',
        'context_auditory', 'context_visual',
        'visual-auditory', 'auditory-visual',
        'colorlessg-wordlist',
        'general-colorlessg',
        'math-nonmath', 'nonmath-math',
        'geometry-othermath',
        'arithmetic_principle-othermath',
        'arithmetic_fact-othermath',
        'theory_of_mind-general', 'context-general', 'theory_of_mind-context',
        'context-theory_of_mind',
        'theory_of_mind_and_context-general']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = dict([(name, con[name]) for name in contrast_names[:16]])
    contrasts['auditory-visual'] =\
        np.sum([con[name] for name in contrast_names[:16:2]], 0) -\
        np.sum([con[name] for name in contrast_names[1:16:2]], 0)
    contrasts['visual-auditory'] = - contrasts['auditory-visual']
    contrasts['colorlessg-wordlist'] =\
        con['colorlessg_auditory'] + con['colorlessg_visual'] - (
        con['wordlist_auditory'] + con['wordlist_visual'])
    contrasts['general-colorlessg'] =\
        con['general_auditory'] + con['general_visual'] -\
        (con['colorlessg_auditory'] + con['colorlessg_visual'])
    contrasts['math-nonmath'] = (
        con['arithmetic_fact_auditory'] + con['arithmetic_fact_visual'] +
        con['arithmetic_principle_auditory'] +
        con['arithmetic_principle_visual'] +
        con['geometry_fact_auditory'] + con['geometry_fact_visual']) - (
        con['theory_of_mind_auditory'] + con['theory_of_mind_visual'] +
        con['context_auditory'] + con['context_visual'] +
        con['general_auditory'] + con['general_visual'])
    contrasts['nonmath-math'] = - contrasts['math-nonmath']
    contrasts['geometry-othermath'] =\
        con['geometry_fact_auditory'] + con['geometry_fact_visual'] - 0.5 * (
        con['arithmetic_fact_auditory'] + con['arithmetic_fact_visual'] +
        con['arithmetic_principle_auditory'] +
        con['arithmetic_principle_visual'])
    contrasts['arithmetic_principle-othermath'] =\
        con['arithmetic_principle_auditory'] +\
        con['arithmetic_principle_visual'] - 0.5 * (
        con['arithmetic_fact_auditory'] + con['arithmetic_fact_visual'] +
        con['geometry_fact_auditory'] + con['geometry_fact_visual'])
    contrasts['arithmetic_fact-othermath'] =\
        con['arithmetic_fact_auditory'] + con['arithmetic_fact_visual'] -\
        0.5 * (
        con['geometry_fact_auditory'] + con['geometry_fact_visual'] +
        con['arithmetic_principle_auditory'] +
        con['arithmetic_principle_visual'])
    contrasts['theory_of_mind-general'] =\
        con['theory_of_mind_auditory'] + con['theory_of_mind_visual'] - (
        con['general_auditory'] + con['general_visual'])
    contrasts['context-general'] =\
        con['context_auditory'] + con['context_visual'] - (
        con['general_auditory'] + con['general_visual'])
    contrasts['theory_of_mind-context'] =\
        con['theory_of_mind_auditory'] + con['theory_of_mind_visual'] - (
        con['context_auditory'] + con['context_visual'])
    contrasts['context-theory_of_mind'] = - contrasts['theory_of_mind-context']
    contrasts['theory_of_mind_and_context-general'] = .5 * (
        con['theory_of_mind_auditory'] + con['theory_of_mind_visual'] +
        con['context_auditory'] + con['context_visual']) - (
        con['general_auditory'] + con['general_visual'])
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def spatial_navigation(design_matrix_columns):
    """ Contrasts for spatial navigation task"""
    #
    contrast_names = [
        'experimental-intersection', 'experimental-control',
        # 'encoding_phase',
        'intersection',
        'retrieval',
        'control', 'pointing_control',
        'experimental', 'pointing_experimental',
        'navigation',
    ]
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)

    contrasts = {#'encoding_phase': con['encoding_phase'],
                 'navigation': con['navigation'],
                 'experimental': con['experimental'],
                 'pointing_experimental': con['pointing_experimental'],
                 'control': con['control'],
                 'pointing_control': con['pointing_control'],
                 'intersection': con['intersection'],
                 'experimental-control': con['experimental'] - con['control'],
                 'retrieval':
                     con['experimental'] + con['pointing_experimental'] -
                     con['control'] - con['pointing_control'],
    }
    contrasts['experimental-intersection'] = (
        contrasts['experimental'] - contrasts['intersection'] / 3)

    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def wedge(design_matrix_columns):
    """ Contrasts for wedge stim"""
    contrast_names = [
        'lower_meridian', 'lower_right', 'right_meridian', 'upper_right',
        'upper_meridian', 'upper_left', 'left_meridian', 'lower_left',
    ]
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = dict([(name, con[name]) for name in contrast_names])
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def ring(design_matrix_columns):
    """ Contrasts for ring stim"""
    contrast_names = ['foveal', 'middle', 'peripheral']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = dict([(name, con[name]) for name in contrast_names])
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def emotional_memory(design_matrix_columns):
    """ Contrasts for emotional memory protocol"""
    contrast_names = ['neutral_image', 'negative_image', 'positive_image', 'object',
                      'positive-neutral_image', 'negative-neutral_image']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = dict([(name, con[name]) for name in contrast_names[:4]])
    contrasts['positive-neutral_image'] = contrasts['positive_image'] - contrasts['neutral_image']
    contrasts['negative-neutral_image'] = contrasts['negative_image'] - contrasts['neutral_image']
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def emotion_recognition(design_matrix_columns):
    """ Contrasts for emotion recognition protocol"""
    contrast_names = [
        'neutral_male', 'angry_male', 'neutral_female', 'angry_female',
        'neutral', 'angry', 'angry-neutral', 'neutral-angry', 'male-female',
        'female-male']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = {
        'neutral_male': con['neutral_male'],
        'angry_male': con['angry_male'],
        'neutral_female': con['neutral_female'],
        'angry_female': con['angry_female'],
        'neutral': con['neutral_male'] + con['neutral_female'],
        'angry': con['angry_male'] + con['angry_female'],
        }
    contrasts['angry-neutral'] = contrasts['angry'] - contrasts['neutral']
    contrasts['neutral-angry'] = - contrasts['angry-neutral']
    contrasts['male-female'] = (
        con['neutral_male'] + con['angry_male'] - con['neutral_female']
        - con['angry_female'])
    contrasts['female-male'] = - contrasts['male-female']
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def oddball(design_matrix_columns):
    """ Contrasts for oddball protocol"""
    contrast_names = ['easy', 'hard', 'hard-easy']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = {
        'easy': con['easy'],
        'hard': con['hard'],
        'hard-easy': con['hard'] - con['easy'],}
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def stop_nogo(design_matrix_columns):
    """ Contrasts for stop nogo protocol"""
    contrast_names = ['go', 'nogo', 'successful_stop', 'unsuccessful_stop',
                      'nogo-go', 'unsuccessful-successful_stop',
                      'successful+nogo-unsuccessful']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = {
        'go': con['go'],
        'nogo': con['nogo'],
        'successful_stop': con['successful_stop'],
        'unsuccessful_stop': con['unsuccessful_stop'],
        'nogo-go': con['nogo'] - con['go'],
        'unsuccessful-successful_stop': con['unsuccessful_stop'] - con['successful_stop'],
        'successful+nogo-unsuccessful': con['successful_stop'] + con['nogo'] -  con['unsuccessful_stop']
    }
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def finger_tapping(design_matrix_columns):
    """ Contrasts for finger tapping protocol"""
    contrast_names = ['specified', 'chosen', 'null',
                      'chosen-specified', 'specified-null', 'chosen-null']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = {
        'specified': con['specified'],
        'chosen': con['chosen'],
        'null': con['null'],
        'chosen-specified': con['chosen'] - con['specified'],
        'specified-null': con['specified'] - con['null'],
        'chosen-null': con['chosen'] - con['null']}
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def vstmc(design_matrix_columns):
    """ Contrasts for vstmc protocol"""
    contrast_names = ['stim_load1', 'stim_load2', 'stim_load3',
                      'resp_load1', 'resp_load2', 'resp_load3',
                      'stim', 'resp', 'stim_load3-load1',
                      'resp_load3-load1'
    ]
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = {
        'stim_load1': con['stim_load1'],
        'stim_load2': con['stim_load2'],
        'stim_load3': con['stim_load3'],
        'resp_load1': con['resp_load1'],
        'resp_load2': con['resp_load2'],
        'resp_load3': con['resp_load3'],
        'stim': con['stim_load1'] + con['stim_load2'] + con['stim_load3'],
        'resp': con['resp_load1'] + con['resp_load2'] + con['resp_load3'],
        'stim_load3-load1': con['stim_load3'] - con['stim_load1'],
        'resp_load3-load1': con['resp_load3'] - con['resp_load1'],
    }
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts



def biological_motion1(design_matrix_columns):
    """ Contrasts for biological motion 1 protocol"""
    contrast_names = ['global_upright', 'global_inverted',
                      'natural_upright', 'natural_inverted',
                      'global_upright - natural_upright',
                      'global_upright - global_inverted',
                      'natural_upright - natural_inverted',
                      'global-natural', 'natural-global',
                      'inverted-upright'
                      ]
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = dict([(name, con[name]) for name in contrast_names[:4]])
    contrasts['global_upright - natural_upright'] =\
        contrasts['global_upright'] - contrasts['natural_upright']
    contrasts['global_upright - global_inverted'] = \
        contrasts['global_upright'] - contrasts['global_inverted']
    contrasts['natural_upright - natural_inverted'] =\
        contrasts['natural_upright'] - contrasts['natural_inverted']
    contrasts['global-natural'] =\
        con['global_upright'] + con['global_inverted'] -\
        con['natural_upright'] - con['natural_inverted']
    contrasts['natural-global'] = - contrasts['global-natural']
    contrasts['inverted-upright'] =\
        - con['global_upright'] + con['global_inverted'] -\
        con['natural_upright'] + con['natural_inverted']
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def biological_motion2(design_matrix_columns):
    """ Contrasts for biological motion 1 protocol"""
    contrast_names = ['modified_upright', 'modified_inverted',
                      'natural_upright', 'natural_inverted',
                      'natural_upright - modified_upright',
                      'modified_upright - modified_inverted',
                      'natural_upright - natural_inverted',
                      'modified-natural', 'natural-modified',
                      'inverted-upright'
                      ]
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = dict([(name, con[name]) for name in contrast_names[:4]])
    contrasts['natural_upright - modified_upright'] =\
        contrasts['natural_upright'] - contrasts['modified_upright']
    contrasts['modified_upright - modified_inverted'] = \
        contrasts['modified_upright'] - contrasts['modified_inverted']
    contrasts['natural_upright - natural_inverted'] =\
        contrasts['natural_upright'] - contrasts['natural_inverted']
    contrasts['modified-natural'] =\
        con['modified_upright'] + con['modified_inverted'] -\
        con['natural_upright'] - con['natural_inverted']
    contrasts['natural-modified'] = - contrasts['modified-natural']
    contrasts['inverted-upright'] =\
        - con['modified_upright'] + con['modified_inverted'] -\
        con['natural_upright'] + con['natural_inverted']
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def dot_patterns(design_matrix_columns):
    """ Contrasts for Stanford's dot patterns protocol"""
    contrast_names = [
        'cue',
        'correct_cue_correct_probe',
        'correct_cue_incorrect_probe',
        'incorrect_cue_correct_probe',
        'incorrect_cue_incorrect_probe',
        'correct_cue_incorrect_probe-correct_cue_correct_probe',
        'incorrect_cue_incorrect_probe-incorrect_cue_correct_probe',
        'correct_cue_incorrect_probe-incorrect_cue_correct_probe',
        'incorrect_cue_incorrect_probe-correct_cue_incorrect_probe',
        'correct_cue-incorrect_cue',
        'incorrect_probe-correct_probe'
    ]
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = {
        'cue': con['cue'],
        'correct_cue_correct_probe': con['correct_cue_correct_probe'],
        'correct_cue_incorrect_probe': con['correct_cue_incorrect_probe'],
        'incorrect_cue_correct_probe': con['incorrect_cue_correct_probe'],
        'incorrect_cue_incorrect_probe': con['incorrect_cue_incorrect_probe'],
        'correct_cue_incorrect_probe-correct_cue_correct_probe':
            con['correct_cue_incorrect_probe'] -
            con['correct_cue_correct_probe'],
        'incorrect_cue_incorrect_probe-incorrect_cue_correct_probe':
            con['incorrect_cue_incorrect_probe'] -
            con['incorrect_cue_correct_probe'],
        'correct_cue_incorrect_probe-incorrect_cue_correct_probe':
            con['correct_cue_incorrect_probe'] -
            con['incorrect_cue_correct_probe'],
        'incorrect_cue_incorrect_probe-correct_cue_incorrect_probe':
            con['incorrect_cue_incorrect_probe'] -
            con['correct_cue_incorrect_probe'],
        'correct_cue-incorrect_cue':
            con['correct_cue_correct_probe']
            + con['correct_cue_incorrect_probe']
            - con['incorrect_cue_correct_probe']
            - con['incorrect_cue_incorrect_probe'],
        'incorrect_probe-correct_probe':
            - con['correct_cue_correct_probe']
            + con['correct_cue_incorrect_probe']
            - con['incorrect_cue_correct_probe']
            + con['incorrect_cue_incorrect_probe'],
    }

    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def columbia_cards(design_matrix_columns):
    """ Contrasts for Stanford's Columbia Cards protocol"""
    contrast_names = ['num_loss_cards', 'loss', 'gain']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = dict([(name, con[name]) for name in contrast_names])
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def discount(design_matrix_columns):
    """ Contrasts for Stanford's discount protocol"""
    contrast_names = ['delay', 'amount']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = dict([(name, con[name]) for name in contrast_names])
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def towertask(design_matrix_columns):
    """ Contrasts for Stanford's Tower task protocol"""
    contrast_names = ['planning_ambiguous_intermediate',
                      'planning_ambiguous_direct',
                      'planning_unambiguous_intermediate',
                      'planning_unambiguous_direct',
                      'move_ambiguous_intermediate',
                      'move_ambiguous_direct',
                      'move_unambiguous_intermediate',
                      'move_unambiguous_direct',
                      'intermediate-direct',
                      'ambiguous-unambiguous']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = dict([(cn, con[cn]) for cn in contrast_names[:-2]])
    contrasts['intermediate-direct'] =\
        con['planning_ambiguous_intermediate']\
        + con['planning_unambiguous_intermediate']\
        - (con['planning_ambiguous_direct']
           + con['planning_unambiguous_direct'])
    contrasts['ambiguous-unambiguous'] =\
        con['planning_ambiguous_intermediate']\
        - con['planning_unambiguous_intermediate'] +\
        con['planning_ambiguous_direct']\
        - con['planning_unambiguous_direct']
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def two_by_two(design_matrix_columns):
    """ Contrasts for Stanford's two-by-two task protocol"""
    contrast_names = [
        'cue_taskstay_cuestay',
        'cue_taskstay_cueswitch',
        'cue_taskswitch_cuestay',
        'cue_taskswitch_cueswitch',
        'stim_taskstay_cuestay',
        'stim_taskstay_cueswitch',
        'stim_taskswitch_cuestay',
        'stim_taskswitch_cueswitch',
        'task_switch-stay',
        'cue_switch-stay']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = dict([(cn, con[cn]) for cn in contrast_names[:-2]])
    contrasts['task_switch-stay'] =\
        con['cue_taskswitch_cueswitch'] + con['cue_taskswitch_cuestay']\
        - con['cue_taskstay_cueswitch'] - con['cue_taskstay_cuestay']
    contrasts['cue_switch-stay'] = con['cue_taskstay_cueswitch']\
        - con['cue_taskstay_cuestay']

    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def attention(design_matrix_columns):
    """ Contrasts for Stanford's attention protocol"""
    contrast_names = [
        'spatial_cue-double_cue',
        'spatial_cue', 'double_cue',
        'incongruent-congruent', 'spatial_incongruent-spatial_congruent',
        'double_incongruent-double_congruent', 'spatial_incongruent',
        'double_congruent', 'spatial_congruent',
        'double_incongruent'
        ]
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = {
        'spatial_cue-double_cue': con['spatialcue'] - con['doublecue'],
        'spatial_cue': con['spatialcue'],
        'double_cue': con['doublecue'],
        'incongruent-congruent':
            con['spatial_incongruent'] - con['spatial_congruent'] +
            con['double_incongruent'] - con['double_congruent'],
        'spatial_incongruent-spatial_congruent':
            con['spatial_incongruent'] - con['spatial_congruent'],
        'double_incongruent-double_congruent':
            con['double_incongruent'] - con['double_congruent'],
        'spatial_incongruent': con['spatial_incongruent'],
        'double_congruent': con['double_congruent'],
        'spatial_congruent': con['spatial_congruent'],
        'double_incongruent': con['double_incongruent']
    }
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def selective_stop_signal(design_matrix_columns):
    """ Contrasts for Stanford's selective_stop_signal protocol"""
    contrast_names = ['go_critical', 'go_noncritical', 'stop', 'ignore',
                      'go_critical-stop', 'go_noncritical-ignore',
                      'stop-ignore', 'ignore-stop']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = {
        'go_critical': con['go_critical'],
        'go_noncritical': con['go_noncritical'],
        'stop': con['stop'],
        'ignore': con['ignore'],
        'go_critical-stop': con['go_critical'] - con['stop'],
        'go_noncritical-ignore': con['go_noncritical'] - con['ignore'],
        'ignore-stop': con['ignore'] - con['stop'],
        'stop-ignore': con['stop'] - con['ignore']
    }
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def stop_signal(design_matrix_columns):
    """Contrasts for the Stanford stop signal protocol"""
    contrast_names = ['go', 'stop', 'stop-go']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = {
        'go': con['go'],
        'stop': con['stop'],
        'stop-go': con['stop'] - con['go'],
    }
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def stroop(design_matrix_columns):
    """Contrasts for the stanford stroop protocol"""
    contrast_names = ['congruent', 'incongruent', 'incongruent-congruent']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = {
        'congruent': con['congruent'],
        'incongruent': con['incongruent'],
        'incongruent-congruent': con['incongruent'] - con['congruent'],
    }
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def lyon_lec2(design_matrix_columns):
    """Contrasts for the lyon lec2 protocol"""
    contrast_names = ['attend', 'unattend', 'attend-unattend']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = {
        'attend': con['attend'],
        'unattend': con['unattend'],
        'attend-unattend': con['attend'] - con['unattend'],
    }
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def lyon_audi(design_matrix_columns):
    """Contrasts for the lyon audi protocol"""
    contrast_names = ['tear', 'suomi', 'yawn', 'human', 'music',
                      'reverse', 'speech', 'alphabet', 'cough', 'environment',
                      'laugh', 'animals',  'silence', 'tear-silence',
                      'suomi-silence',
                      'yawn-silence', 'human-silence', 'music-silence',
                      'reverse-silence', 'speech-silence', 'alphabet-silence',
                      'cough-silence', 'environment-silence',
                      'laugh-silence', 'animals-silence']

    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = dict([(name, con[name]) for name in contrast_names[:13]])
    for name in contrast_names[:12]:
        contrasts[name + '-silence'] = con[name] - con['silence']
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def lyon_visu(design_matrix_columns):
    """Contrasts for the lyon visu protocol"""
    contrast_names = ['scrambled', 'scene', 'tool', 'face', 'target_fruit',
                      'house', 'animal', 'characters', 'pseudoword',
                      'scene-scrambled', 'tool-scrambled',
                      'face-scrambled', 'house-scrambled', 'animal-scrambled',
                      'characters-scrambled', 'pseudoword-scrambled', ]
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    canonical_contrasts = ['scrambled', 'scene', 'tool', 'face',
                           'house', 'animal', 'characters', 'pseudoword']
    contrast = dict([(name, con[name]) for name in canonical_contrasts])
    # average = np.array([x for x in contrast.values()]).sum(0) * 1. / 8
    contrast['target_fruit'] = con['target_fruit']
    contrast['scene-scrambled'] = contrast['scene'] - contrast['scrambled']
    contrast['tool-scrambled'] = contrast['tool'] - contrast['scrambled']
    contrast['face-scrambled'] = contrast['face'] - contrast['scrambled']
    contrast['house-scrambled'] = contrast['house'] - contrast['scrambled']
    contrast['animal-scrambled'] = contrast['animal'] - contrast['scrambled']
    contrast['characters-scrambled'] =\
        contrast['characters'] - contrast['scrambled']
    contrast['pseudoword-scrambled'] =\
        contrast['pseudoword'] - contrast['scrambled']
    assert((sorted(contrast.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrast)
    _append_effects_interest_contrast(design_matrix_columns, contrast)
    return contrast


def lyon_lec1(design_matrix_columns):
    """Contrasts for the lec1 protocol"""
    contrast_names = ['pseudoword', 'word', 'random_string', 'word-pseudoword',
                      'word-random_string', 'pseudoword-random_string',
                      ]
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = {
        'pseudoword': con['pseudoword'],
        'word': con['word'],
        'random_string': con['random_string'],
        'word-pseudoword': con['word'] - con['pseudoword'],
        'word-random_string': con['word'] - con['random_string'],
        'pseudoword-random_string': con['pseudoword'] - con['random_string']
    }
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def audio(design_matrix_columns):
    """Contrasts for the audio protocol"""
    contrast_names = [
        'animal', 'music', 'nature',
        'speech', 'tool', 'voice',
        'animal-others', 'music-others', 'nature-others',
        'speech-others', 'tool-others', 'voice-others',
        'mean-silence',
        'animal-silence', 'music-silence', 'nature-silence',
        'speech-silence', 'tool-silence', 'voice-silence',
        ]
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    others = (con['animal'] + con['music'] + con['nature'] +
              con['speech'] + con['tool'] + con['voice']) / 6
    contrasts = {
        'animal': con['animal'],
        'music': con['music'],
        'nature': con['nature'],
        'speech': con['speech'],
        'tool': con['tool'],
        'voice': con['voice'],
        'mean-silence': others - con['silence'],
        'animal-others': con['animal'] - others,
        'music-others': con['music'] - others,
        'nature-others': con['nature'] - others,
        'speech-others': con['speech'] - others,
        'tool-others': con['tool'] - others,
        'voice-others': con['voice'] - others,
        'animal-silence': con['animal'] - con['silence'],
        'music-silence': con['music'] - con['silence'],
        'nature-silence': con['nature'] - con['silence'],
        'speech-silence': con['speech'] - con['silence'],
        'tool-silence': con['tool'] - con['silence'],
        'voice-silence': con['voice'] - con['silence'],
        }
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def lyon_mveb(design_matrix_columns):
    """ Contrasts for Lyon mveb localizer"""
    contrast_names = [
        'letter_occurrence_response', '2_letters_different', '2_letters_same',
        '4_letters_different', '4_letters_same',
        '6_letters_different', '6_letters_same',
        '2_letters_different-same',
        '4_letters_different-same', '6_letters_different-same',
        '6_letters_different-2_letters_different']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    #
    contrasts = dict([(key, con[key]) for key in contrast_names[1:7]])
    contrasts['letter_occurrence_response'] = con['response']
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
    """ Contrasts for Lyon mvis localizer"""
    contrast_names = ['dot_displacement_response',
                      '2_dots-2_dots_control', '4_dots-4_dots_control',
                      '6_dots-6_dots_control', '6_dots-2_dots', 'dots-control']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    # contrasts = dict([(cname, con[cname]) for cname in contrast_names[:-4]])
    contrasts = {'dot_displacement_response': con['response']}
    contrasts['2_dots-2_dots_control'] = con['2_dots'] - con['2_dots_control']
    contrasts['4_dots-4_dots_control'] = con['4_dots'] - con['4_dots_control']
    contrasts['6_dots-6_dots_control'] = con['6_dots'] - con['6_dots_control']
    contrasts['6_dots-2_dots'] = con['6_dots'] - con['2_dots']
    contrasts['dots-control'] = con['6_dots'] + con['4_dots'] + con['2_dots']\
        - (con['2_dots_control'] + con['6_dots_control'] +
           con['4_dots_control'])
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def lyon_moto(design_matrix_columns):
    """ Contrasts for Lyon motor localizer"""
    contrast_names = [
        'instructions', 'finger_right-fixation', 'finger_left-fixation',
        'foot_left-fixation', 'foot_right-fixation', 'hand_left-fixation',
        'hand_right-fixation', 'saccade-fixation', 'tongue-fixation']
    elementary_contrasts = [
        'foot_left', 'foot_right', 'finger_right', 'finger_left',
        'saccade_left', 'saccade_right', 'hand_left', 'hand_right',
        'fixation_right', 'tongue_right', 'fixation_left',  'tongue_left']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    # avg = np.mean([con[cname] for cname in elementary_contrasts], 0)
    contrasts = {'instructions': con['instructions']}
    con['fixation'] = .5 * (con['fixation_left'] + con['fixation_right'])
    contrasts['finger_right-fixation'] = con['finger_right'] - con['fixation']
    contrasts['finger_left-fixation'] = con['finger_left'] - con['fixation']
    contrasts['foot_left-fixation'] = con['foot_left'] - con['fixation']
    contrasts['foot_right-fixation'] = con['foot_right'] - con['fixation']
    contrasts['hand_left-fixation'] = con['hand_left'] - con['fixation']
    contrasts['hand_right-fixation'] = con['hand_right'] - con['fixation']
    contrasts['saccade-fixation'] = con['saccade_left'] + con['saccade_right']\
        - 2 * con['fixation']
    contrasts['tongue-fixation'] = con['tongue_left'] + con['tongue_right']\
        - 2 * con['fixation']
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def lyon_mcse(design_matrix_columns):
    """ Contrasts for Lyon MCSE localizer"""
    contrast_names = [
        'high_salience_left', 'high_salience_right',
        'low_salience_left', 'low_salience_right',
        'high-low_salience', 'low-high_salience',
        'salience_left-right', 'salience_right-left',
        'low+high_salience']
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
        'low+high_salience':
            con['hi_salience_left'] + con['hi_salience_right']
            + con['low_salience_left'] + con['low_salience_right'],
    }
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def bang(design_matrix_columns):
    """ Contrasts for bang experiment"""
    contrast_names = ['talk', 'no_talk', 'talk-no_talk']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = {
        'talk': con['talk'],
        'no_talk': con['no_talk'],
        'talk-no_talk': con['talk'] - con['no_talk'],}
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def self_localizer(design_matrix_columns):
    """ Contrasts for self experiment"""
    contrast_names = [
        'encode_self-other', 'encode_other', 'encode_self',
        'instructions', 'false_alarm', 'correct_rejection',
        'recognition_hit', 'recognition_hit-correct_rejection',
        'recognition_self-other', 'recognition_self_hit',
        'recognition_other_hit'
    ]
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)

    try:
        recognition_hit = con['recognition_other_hit'] +\
            con['recognition_self_hit']
    except KeyError:
        if 'recognition_self_hit' in con.keys():
            recognition_hit = con['recognition_self_hit']
        elif 'recognition_other_hit' in con.keys():
            recognition_hit = con['recognition_other_hit']
        else:
            recognition_hit = con['recognition_other_no_response']
    try:
        correct_rejection = con['correct_rejection']
    except KeyError:
        correct_rejection = con['false_alarm']  #
    try:
        recognition_self_hit = con['recognition_self_hit']
    except KeyError:
        recognition_self_hit = con['recognition_self_miss']
    try:
        recognition_self = con['recognition_self_hit'] +\
            con['recognition_self_miss']
    except KeyError:
        if 'recognition_self_miss' in con.keys():
            recognition_self = con['recognition_self_miss']
        else:
            recognition_self = con['recognition_self_hit']
    try:
        recognition_other = con['recognition_other_hit'] +\
            con['recognition_other_miss']
    except KeyError:
        if 'recognition_other_hit' in con.keys():
            recognition_other = con['recognition_other_hit']
        else:
            recognition_other = con['recognition_other_miss']
    try:
        recognition_other_hit = con['recognition_other_hit']
    except KeyError:
        recognition_other_hit = con['recognition_other_miss']

    contrasts = {
        'encode_self-other': con['encode_self'] - con['encode_other'],
        'encode_other': con['encode_other'],
        'encode_self': con['encode_self'],
        'instructions': con['instructions'],
        'false_alarm': con['false_alarm'],
        'recognition_hit': recognition_hit,
        'recognition_self_hit': recognition_self_hit,
        'recognition_hit-correct_rejection':
            recognition_hit - correct_rejection,
        'correct_rejection': correct_rejection,
        'recognition_self-other': recognition_self - recognition_other,
        'recognition_other_hit': recognition_other_hit,
        }

    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def vstm(design_matrix_columns):
    """ contrasts for vstm task, Knops protocol"""
    contrast_names = [
       'vstm_linear',
       'vstm_constant',
       'vstm_quadratic']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    constant = np.ones(6)
    linear = np.linspace(-1, 1, 6)
    quadratic = linear ** 2 - (linear ** 2).mean()
    con = _elementary_contrasts(design_matrix_columns)
    response = np.array([con['response_num_%d' % i]
                        for i in range(1, 7)])
    contrasts = {
        'vstm_constant': np.dot(constant, response),
        'vstm_linear': np.dot(linear, response),
        'vstm_quadratic': np.dot(quadratic, response),
    }
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def enumeration(design_matrix_columns):
    """ contrasts for vstm task, Knops protocol"""
    contrast_names = [
        'enumeration_linear',
        'enumeration_constant',
        'enumeration_quadratic']

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
        'enumeration_constant': np.dot(constant, response),
        'enumeration_linear': np.dot(linear, response),
        'enumeration_quadratic': np.dot(quadratic, response),
    }
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def pain_movie(design_matrix_columns):
    """ Contrast for pain task, TOM protocol"""
    contrast_names = ['movie_pain', 'movie_mental', 'movie_mental-pain',]
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = {'movie_pain': con['pain'],
                 'movie_mental': con['mental'],
                 'movie_mental-pain': con['mental'] - con['pain'],
                 }
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def emotional_pain(design_matrix_columns):
    """ Contrast for pain task, TOM protocol"""
    contrast_names = ['physical_pain', 'emotional_pain',
                      'emotional-physical_pain']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = {'emotional_pain': con['emotional_pain'],
                 'physical_pain': con['physical_pain'],
                 'emotional-physical_pain':
                 con['emotional_pain'] - con['physical_pain'],
                 }
    assert((sorted(contrasts.keys()) == sorted(contrast_names)))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts


def theory_of_mind(design_matrix_columns):
    """ Contrast for tom task, TOM protocol"""
    contrast_names = ['belief', 'photo', 'belief-photo']
    if design_matrix_columns is None:
        return dict([(name, []) for name in contrast_names])
    con = _elementary_contrasts(design_matrix_columns)
    contrasts = {'photo': con['photo'],
                 'belief': con['belief'],
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


def mtt_we_relative(design_matrix_columns):
    """Contrast for MTT west-east task, relative setting"""
    contrast_list = [
        'we_average_reference',
        'we_all_space_cue',
        'we_all_time_cue',
        'we_westside_event',
        'we_eastside_event',
        'we_before_event',
        'we_after_event',
        'we_all_event_response',
        'we_all_space-time_cue',
        'we_all_time-space_cue',
        'we_average_event',
        'we_space_event',
        'we_time_event',
        'we_space-time_event',
        'we_time-space_event',
        'westside-eastside_event',
        'eastside-westside_event',
        'we_before-after_event',
        'we_after-before_event'
    ]
    if design_matrix_columns is None:
        return dict([(key, []) for key in contrast_list])

    con = _beta_contrasts(design_matrix_columns)
    contrasts = {
        'we_average_reference': con['we_all_reference'],
        'we_all_space_cue': con['we_all_space_cue'],
        'we_all_time_cue': con['we_all_time_cue'],
        'we_westside_event':
            con['we_westside_close_event']
            + con['we_westside_far_event'],
        'we_eastside_event':
            con['we_eastside_close_event']
            + con['we_eastside_far_event'],
        'we_before_event':
            con['we_before_close_event']
            + con['we_before_far_event'],
        'we_after_event':
            con['we_after_close_event']
            + con['we_after_far_event'],
        'we_all_event_response': con['we_all_event_response']}

    contrasts['we_all_space-time_cue'] =\
        contrasts['we_all_space_cue'] - contrasts['we_all_time_cue']
    contrasts['we_all_time-space_cue'] = - contrasts['we_all_space-time_cue']
    contrasts['we_space_event'] =\
        contrasts['we_westside_event'] + contrasts['we_eastside_event']
    contrasts['we_time_event'] =\
        contrasts['we_before_event'] + contrasts['we_after_event']
    contrasts['we_average_event'] =\
        contrasts['we_space_event'] + contrasts['we_time_event']
    contrasts['we_space-time_event'] =\
        contrasts['we_space_event'] - contrasts['we_time_event']
    contrasts['we_time-space_event'] = - contrasts['we_space-time_event']
    contrasts['westside-eastside_event'] =\
        contrasts['we_westside_event'] - contrasts['we_eastside_event']
    contrasts['eastside-westside_event'] =\
        - contrasts['westside-eastside_event']
    contrasts['we_before-after_event'] =\
        contrasts['we_before_event'] - contrasts['we_after_event']
    contrasts['we_after-before_event'] = - contrasts['we_before-after_event']
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts

def mtt_sn_relative(design_matrix_columns):
    """Contrast for MTT south-north task, relative setting"""
    contrast_list = [
        'sn_average_reference',
        'sn_all_space_cue',
        'sn_all_time_cue',
        'sn_southside_event',
        'sn_northside_event',
        'sn_before_event',
        'sn_after_event',
        'sn_all_event_response',
        'sn_all_space-time_cue',
        'sn_all_time-space_cue',
        'sn_average_event',
        'sn_space_event',
        'sn_time_event',
        'sn_space-time_event',
        'sn_time-space_event',
        'northside-southside_event',
        'southside-northside_event',
        'sn_before-after_event',
        'sn_after-before_event'
    ]
    if design_matrix_columns is None:
        return dict([(key, []) for key in contrast_list])

    con = _beta_contrasts(design_matrix_columns)
    contrasts = {
        'sn_average_reference': con['sn_all_reference'],
        'sn_all_space_cue': con['sn_all_space_cue'],
        'sn_all_time_cue': con['sn_all_time_cue'],
        'sn_southside_event':
            con['sn_southside_close_event']
            + con['sn_southside_far_event'],
        'sn_northside_event':
            con['sn_northside_close_event']
            + con['sn_northside_far_event'],
        'sn_before_event':
            con['sn_before_close_event']
            + con['sn_before_far_event'],
        'sn_after_event':
            con['sn_after_close_event']
            + con['sn_after_far_event'],
        'sn_all_event_response': con['sn_all_event_response']}

    contrasts['sn_all_space-time_cue'] =\
        contrasts['sn_all_space_cue'] - contrasts['sn_all_time_cue']
    contrasts['sn_all_time-space_cue'] = - contrasts['sn_all_space-time_cue']
    contrasts['sn_space_event'] =\
        contrasts['sn_southside_event'] + contrasts['sn_northside_event']
    contrasts['sn_time_event'] =\
        contrasts['sn_before_event'] + contrasts['sn_after_event']
    contrasts['sn_average_event'] =\
        contrasts['sn_space_event'] + contrasts['sn_time_event']
    contrasts['sn_space-time_event'] =\
        contrasts['sn_space_event'] - contrasts['sn_time_event']
    contrasts['sn_time-space_event'] = - contrasts['sn_space-time_event']
    contrasts['southside-northside_event'] =\
        contrasts['sn_southside_event'] - contrasts['sn_northside_event']
    contrasts['northside-southside_event'] =\
        - contrasts['southside-northside_event']
    contrasts['sn_before-after_event'] =\
        contrasts['sn_before_event'] - contrasts['sn_after_event']
    contrasts['sn_after-before_event'] = - contrasts['sn_before-after_event']
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
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
        'word-consonant_string', 'jabberwocky-pseudo',
        'word-pseudo', 'pseudo-consonant_string',
        'sentence-consonant_string', 'simple-consonant_string',
        'complex-consonant_string', 'sentence-pseudo', 'probe',
        'jabberwocky-consonant_string']
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
        'jabberwocky-pseudo': con['jabberwocky'] - con['pseudoword_list'],
        'jabberwocky-consonant_string':
            con['jabberwocky'] - con['consonant_strings'],
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
    assert(sorted(contrasts.keys()) == sorted(contrast_names))
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

    assert(sorted(contrasts.keys()) == sorted(contrast_names))
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
    assert(sorted(contrasts.keys()) == sorted(contrast_names))
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
    assert(sorted(contrasts.keys()) == sorted(contrast_names))
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
    assert(sorted(contrasts.keys()) == sorted(contrast_names))
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
    assert(sorted(contrasts.keys()) == sorted(contrast_names))
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
    assert(sorted(contrasts.keys()) == sorted(contrast_names))
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
    assert(sorted(contrasts.keys()) == sorted(contrast_names))
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
    assert(sorted(contrasts.keys()) == sorted(contrast_names))
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
    assert(sorted(contrasts.keys()) == sorted(contrast_names))
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
    assert(sorted(contrasts.keys()) == sorted(contrast_names))
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
    assert(sorted(contrasts.keys()) == sorted(contrast_names))
    _append_derivative_contrast(design_matrix_columns, contrasts)
    _append_effects_interest_contrast(design_matrix_columns, contrasts)
    return contrasts
