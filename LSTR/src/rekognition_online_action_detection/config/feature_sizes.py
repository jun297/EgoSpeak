# dimension of features extracted from different models
# used in feature_head, dataset, models

FEATURE_SIZES = {
    'rgb_anet_resnet50': 2048,
    'flow_anet_resnet50': 2048,
    'rgb_kinetics_bninception': 1024,
    'flow_kinetics_bninception': 1024,
    'rgb_farl': 512,
    'rgb_kinetics_resnet50': 2048,
    'rgb_kinetics_resnet50_debug': 2048,
    'rgb_kinetics_resnet50_debug_resized': 2048,
    'flow_kinetics_resnet50': 2048,
    'flow_nv_kinetics_bninception': 1024,
    'rgb_kinetics_i3d': 2048,
    'flow_kinetics_i3d': 2048,
    'audio_wav2vec2': 5120,
    'audio_wav2vec2_left': 5120,
    'audio_wav2vec2_right': 5120,
}