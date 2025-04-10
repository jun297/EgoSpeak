# Evaluation code from LSTR

from collections import OrderedDict
from multiprocessing import Pool

import numpy as np
from sklearn.metrics import average_precision_score


def calibrated_average_precision_score(y_true, y_score):
    """Compute calibrated average precision (cAP), which is particularly
    proposed for the TVSeries dataset.
    """
    y_true_sorted = y_true[np.argsort(-y_score)]
    tp = y_true_sorted.astype(float)
    fp = np.abs(y_true_sorted.astype(float) - 1)
    tps = np.cumsum(tp)
    fps = np.cumsum(fp)
    ratio = np.sum(tp == 0) / np.sum(tp)
    cprec = tps / (tps + fps / (ratio + np.finfo(float).eps) + np.finfo(float).eps)
    cap = np.sum(cprec[tp == 1]) / np.sum(tp)
    return cap

def perframe_average_precision(prediction, ground_truth, class_names, ignore_index,
                               postprocessing=None, metrics='AP'):
    """Compute (frame-level) average precision between ground truth and
    predictions data frames.
    """
    result = OrderedDict()
    ground_truth = np.array(ground_truth)
    prediction = np.array(prediction)

    # Postprocessing
    if postprocessing is not None:
        ground_truth, prediction = postprocessing(ground_truth, prediction)
        
    # Build metrics
    if metrics == 'AP':
        compute_score = average_precision_score
    elif metrics == 'cAP':
        # print('cAP')
        compute_score = calibrated_average_precision_score
    else:
        raise RuntimeError('Unknown metrics: {}'.format(metrics))

    # Ignore backgroud class
    ignore_set = set([ignore_index])
    
    # print('ignore_set = set([{ignore_index}])')
    
    # Enable background according to ignore_index
    # ignore_set = set([ignore_index])
    
    # Compute average precision
    result['per_class_AP'] = OrderedDict()
    result['num'] = OrderedDict()
    print(f"NUM FRAMES: {np.sum(ground_truth[:, 1:])}")
    for idx, class_name in enumerate(class_names):
        if idx not in ignore_set:
            # ground_truth.shape [2901376, 2]
            # prediction.shape [45334 ?]
            if np.any(ground_truth[:, idx]):
                ap_score = compute_score(ground_truth[:, idx], prediction[:, idx])
                result['per_class_AP'][class_name] = ap_score
                result['num'][class_name] = f'[true: {int(np.sum(ground_truth[:, idx]))}, pred:{int(np.sum(prediction[:,idx]))}, AP:{ap_score*100:.1f}]'
    result['mean_AP'] = np.mean(list(result['per_class_AP'].values()))
    print(f"Per_class_AP: {result['per_class_AP']}")
    return result

def event_accuracy(pred_scores_dict, gt_targets_dict):
    th_session_accuracies = {}
    
    threshold = 0.5
    # ERROR: something is wrong
    # PLEASE EDIT:
    # pred_scores_dict here has wrong dimension
    # [64, 128, 10, 3] <- this is because I am using trainloader (do not evaluate on trainloader)
    for threshold in np.arange(0.1, 1.0, 0.2): # 0.1, 0.3, 0.5, 0.7, 0.9
        num_events = 0
        th_total_speaking_start_correct = 0
        th_total_speaking_end_correct = 0
        
        for session in pred_scores_dict.keys():       
            th_session_speaking_start_correct = 0
            th_session_speaking_start_predictions = 0
            
            th_session_speaking_end_correct = 0
            th_session_speaking_end_predictions = 0
            
            argmax_session_speaking_start_correct = 0
            argmax_session_speaking_end_correct = 0
            
            pred_scores = pred_scores_dict[session]
            # pred_scores: [num_feature_frames, anticipation_steps, num_classes]
            # classes: background, target speaker speaking, other speaker speaking
            
            # 12/06: To-do
            # I forgot whether miniroad AA step 0 means anticipate 1 step future or current
            # I did change something it would likely to be 
            ant_length = pred_scores.shape[1]
            
            gt_targets = gt_targets_dict[session]
            event_labels = np.load(f'/home/junhyeok/projects/turn-taking/datasets/EasyCom/target_perframe/events/{session}.npy')
            
            num_speaking_start = 0
            num_speaking_end = 0
            print(f'{session} loaded, {len(event_labels)} events')
            # 12/06:
            # IndexError: index 8504 is out of bounds for axis 0 with size 8390
            # why the event_idx has much higher value than the length of pred_scores?
            # check how the event label is made  
            
            # event_label: [event_idx, event_class]
            # ['18', 'target speaker speaking start']
            for event_label in event_labels:
                event_idx, event_class = event_label
                event_idx = int(event_idx)
                
                if event_class == 'target speaker speaking start':
                    num_speaking_start += 1
                    num_speaking_end += 1
                    # right before the event start, want to see how the model predict?
                    ant_prob = pred_scores[event_idx-1, 0, :]
                    if np.argmax(ant_prob) == 1:
                        argmax_session_speaking_start_correct += 1
                    
                    ant_target_prob = pred_scores[event_idx-1, 0, 1]
                    if ant_target_prob >= threshold:
                        th_session_speaking_start_correct += 1
                    
                elif event_class == 'target speaker speaking end':
                    # right after the event end, want to see how the model predict?
                    ant_prob = pred_scores[event_idx, 0, :]
                    if np.argmax(ant_prob) == 0:
                        argmax_session_speaking_end_correct += 1
                    
                    # target speaker will not speaker anymore
                    ant_target_prob = pred_scores[event_idx, 0, 1]
                    if ant_target_prob < threshold:
                        th_session_speaking_end_correct += 1
            
            print(f'Threshold {threshold:.2f}) {session} speaking start accuracy: {(th_session_speaking_start_correct / (len(event_labels) / 2)):.4f}')
            print(f'Threshold {threshold:.2f}) {session} speaking end accuracy: {(th_session_speaking_end_correct / (len(event_labels)/ 2 )):.4f}')
            print(f'argmax {session} speaking start accuracy: {argmax_session_speaking_start_correct / (len(event_labels) / 2)}')
            print(f'argmax {session} speaking end accuracy: {argmax_session_speaking_end_correct / (len(event_labels)/ 2 )}')
            th_session_accuracies[session] = [th_session_speaking_start_correct / len(event_labels), th_session_speaking_end_correct / len(event_labels)]
            th_total_speaking_start_correct += th_session_speaking_start_correct
            th_total_speaking_end_correct += th_session_speaking_end_correct
            num_events += len(event_labels)
            
        th_total_speaking_start_accuracy = th_total_speaking_start_correct / len(event_labels)
        th_total_speaking_end_accuracy = th_total_speaking_end_correct / len(event_labels)
        
        th_total_accuracy = (th_total_speaking_start_correct + th_total_speaking_end_correct) / (num_events)
        print(f'total speaking start accuracy: {th_total_accuracy}')
        
    return th_total_accuracy
    
def get_stage_pred_scores(gt_targets, pred_scores, perc_s, perc_e):
    starts = []
    ends = []
    stage_gt_targets = []
    stage_pred_scores = []
    for i in range(len(gt_targets)):
        if gt_targets[i] == 0:
            stage_gt_targets.append(gt_targets[i])
            stage_pred_scores.append(pred_scores[i])
        else:
            if i == 0 or gt_targets[i - 1] == 0:
                starts.append(i)
            if i == len(gt_targets) - 1 or gt_targets[i + 1] == 0:
                ends.append(i)
    if len(starts) != len(ends):
        raise ValueError('starts and ends cannot pair!')

    action_lens = [ends[i] - starts[i] for i in range(len(starts))]
    stage_starts = [starts[i] + int(action_lens[i] * perc_s) for i in range(len(starts))]
    stage_ends = [max(stage_starts[i] + 1, starts[i] + int(action_lens[i] * perc_e)) for i in range(len(starts))]
    for i in range(len(starts)):
        stage_gt_targets.extend(gt_targets[stage_starts[i]: stage_ends[i]])
        stage_pred_scores.extend(pred_scores[stage_starts[i]: stage_ends[i]])
    return np.array(stage_gt_targets), np.array(stage_pred_scores)


def perstage_average_precision(prediction, ground_truth,
                               class_names, postprocessing, 
                               metrics='cAP'):
    result = OrderedDict()
    ground_truth = np.array(ground_truth)
    prediction = np.array(prediction)

    # Postprocessing
    if postprocessing is not None:
        ground_truth, prediction = postprocessing(ground_truth, prediction)

    # Build metrics
    if metrics == 'AP':
        compute_score = average_precision_score
    elif metrics == 'cAP':
        compute_score = calibrated_average_precision_score
    else:
        raise RuntimeError('Unknown metrics: {}'.format(metrics))

    raise NotImplementedError('This function is not implemented yet.')
    # Ignore backgroud class
    # ignore_index = set([0])

    # Compute average precision
    for perc_s in range(10):
        perc_e = perc_s + 1
        stage_name = '{:2}%_{:3}%'.format(perc_s * 10, perc_e * 10)
        result[stage_name] = OrderedDict({'per_class_AP': OrderedDict()})
        for idx, class_name in enumerate(class_names):
            if idx not in ignore_index:
                stage_gt_targets, stage_pred_scores = get_stage_pred_scores(
                    (ground_truth[:, idx] == 1).astype(int),
                    prediction[:, idx],
                    perc_s / 10,
                    perc_e / 10,
                )
                result[stage_name]['per_class_AP'][class_name] = \
                    compute_score(stage_gt_targets, stage_pred_scores)
        result[stage_name]['mean_AP'] = \
            np.mean(list(result[stage_name]['per_class_AP'].values()))

    return result
