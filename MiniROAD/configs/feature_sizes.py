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
    'text_word2vec': 300,
    'text_fasttext_crawl': 300,
    'text_gpt2_last_subword_window_size_0': 768,
    'text_gpt2_last_subword_window_size_20': 768,
    'text_gpt2_last_subword_window_size_50': 768,
    'text_gpt2_last_subword_window_size_100': 768,
}