import shutil

import numpy as np
from sklearn.linear_model import LinearRegression
import ujson as json
import os.path as ops
import os

from tools.generate_prediction_results import generate_prediction_result


class LaneEval(object):
    lr = LinearRegression()
    pixel_thresh = 20
    pt_thresh = 0.85

    @staticmethod
    def get_angle(xs, y_samples):
        xs, ys = xs[xs >= 0], y_samples[xs >= 0]
        if len(xs) > 1:
            LaneEval.lr.fit(ys[:, None], xs)
            k = LaneEval.lr.coef_[0]
            theta = np.arctan(k)
        else:
            theta = 0
        return theta

    @staticmethod
    def line_accuracy(pred, gt, thresh):
        pred = np.array([p if p >= 0 else -100 for p in pred])
        gt = np.array([g if g >= 0 else -100 for g in gt])
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1., 0.)) / len(gt)

    @staticmethod
    def bench(pred, gt, y_samples):
        if any(len(p) != len(y_samples) for p in pred):
            raise Exception('Format of lanes error.')
        # if running_time > 200 or len(gt) + 2 < len(pred):
        #     return 0., 0., 1.
        angles = [LaneEval.get_angle(np.array(x_gts), np.array(y_samples)) for x_gts in gt]
        threshs = [LaneEval.pixel_thresh / np.cos(angle) for angle in angles]
        line_accs = []
        fp, fn = 0., 0.
        matched = 0.
        for x_gts, thresh in zip(gt, threshs):
            accs = [LaneEval.line_accuracy(np.array(x_preds), np.array(x_gts), thresh) for x_preds in pred]
            max_acc = np.max(accs) if len(accs) > 0 else 0.
            if max_acc < LaneEval.pt_thresh:
                fn += 1
            else:
                matched += 1
            line_accs.append(max_acc)
        fp = len(pred) - matched
        if len(gt) > 4 and fn > 0:
            fn -= 1
        s = sum(line_accs)
        if len(gt) > 4:
            s -= min(line_accs)
        return s / max(min(4.0, len(gt)), 1.), fp / len(pred) if len(pred) > 0 else 0., fn / max(min(len(gt), 4.) , 1.)

    @staticmethod
    def bench_one_submit(pred_file, gt_file):
        try:
            json_pred = [json.loads(line) for line in open(pred_file).readlines()]
        except BaseException as e:
            raise Exception('Fail to load json file of the prediction.')
        json_gt = [json.loads(line) for line in open(gt_file).readlines()]
        if len(json_gt) != len(json_pred):
            raise Exception('We do not get the predictions of all the test tasks')
        gts = {l['raw_file']: l for l in json_gt}
        accuracy, fp, fn = 0., 0., 0.
        for pred in json_pred:
            if 'raw_file' not in pred or 'lanes' not in pred:
                raise Exception('raw_file or lanes not in some predictions.')
            raw_file = pred['raw_file']
            pred_lanes = pred['lanes']
            # run_time = pred['run_time']
            if raw_file not in gts:
                raise Exception('Some raw_file from your predictions do not exist in the test tasks.')
            gt = gts[raw_file]
            gt_lanes = gt['lanes']
            y_samples = gt['h_samples']
            try:
                a, p, n = LaneEval.bench(pred_lanes, gt_lanes, y_samples)
            except BaseException as e:
                raise Exception('Format of lanes error.')
            accuracy += a
            fp += p
            fn += n
        num = len(gts)
        # the first return parameter is the default ranking parameter
        return json.dump([
            {'name': 'Accuracy', 'value': accuracy / num, 'order': 'desc'},
            {'name': 'FP', 'value': fp / num, 'order': 'asc'},
            {'name': 'FN', 'value': fn / num, 'order': 'asc'}
        ],open('/home/stevemaary/data/pred/accuracy_result.json','w'))


def compare_pred_label (pred_json_path, label_json_path, dst_dir, readable):
    json_pred = [json.loads(line) for line in open(pred_json_path).readlines()]
    json_label = [json.loads(line) for line in open(label_json_path).readlines()]

    original_files = {line['raw_file']: line for line in json_label}

    missed_all = []
    missed_list_1 = []
    missed_list_2 = []
    missed_list_3 = []
    missed_list_4 = []
    missed_list_5 = []
    more_list = []

    for pred in json_pred:
        raw_file = pred['raw_file']
        pred_lanes = pred['lanes']
        if raw_file not in original_files:
            raise Exception('Some raw_file from your predictions do not exist in the test tasks.')
        label = original_files[raw_file]
        label_lanes = label['lanes']

        result_file_name = pred['result_file']  # .split('/')[-1]

        if len(pred_lanes)==0:
            missed_all.append(result_file_name)
            continue

        missed_lanes = (len(label_lanes)-len(pred_lanes))
        if missed_lanes == 1:
            missed_list_1.append(result_file_name)
        elif missed_lanes  == 2:
            missed_list_2.append(result_file_name)
        elif missed_lanes == 3:
            missed_list_3.append(result_file_name)
        elif missed_lanes  == 4:
            missed_list_4.append(result_file_name)
        elif missed_lanes > 4:
            missed_list_5.append(result_file_name)
        elif missed_lanes < 0:
            more_list.append(result_file_name)
        continue

    with open(dst_dir, 'w') as result:
        data = {
            'no_prediction': missed_all,
            'missed_1': missed_list_1,
            'missed_2': missed_list_2,
            'missed_3': missed_list_3,
            'missed_4': missed_list_4,
            'more_than_4': missed_list_5,
            'more':more_list,
            'count':{
                'no_prediction': len(missed_all),
                'missed_1': len(missed_list_1),
                'missed_2': len(missed_list_2),
                'missed_3': len(missed_list_3),
                'missed_4': len(missed_list_4),
                'more_than_4': len(missed_list_5),
                'more': len(more_list)
            }
        }
        if readable:
            json.dump(data, result, indent=2)
        else:
            json.dump(data,result)

    return


def copy_missed_results(src_dir, dst_dir):
    missed_all_path = ops.join(dst_dir,"missed_all")
    missed_1_path = ops.join(dst_dir, "missed_1")
    missed_2_path = ops.join(dst_dir, "missed_2")
    missed_3_path = ops.join(dst_dir, "missed_3")
    missed_4_path = ops.join(dst_dir, "missed_4")
    missed_more_path = ops.join(dst_dir, "missed_more")
    pred_more_path = ops.join(dst_dir,"pred_more")

    with open(src_dir) as file:
        for line in file:
            info = json.loads(line)
            missed_all_list = info['no_prediction']
            missed_1_list = info['missed_1']
            missed_2_list = info['missed_2']
            missed_3_list = info['missed_3']
            missed_4_list = info['missed_4']
            missed_5_list = info['more_than_4']
            more_list = info['more']

            for path in missed_all_list:
                shutil.copyfile(path, ops.join(missed_all_path, path.split('/')[-1]))
            for path in missed_1_list:
                shutil.copyfile(path, ops.join(missed_1_path, path.split('/')[-1]))
            for path in missed_2_list:
                shutil.copyfile(path, ops.join(missed_2_path, path.split('/')[-1]))
            for path in missed_3_list:
                shutil.copyfile(path, ops.join(missed_3_path, path.split('/')[-1]))
            for path in missed_4_list:
                shutil.copyfile(path, ops.join(missed_4_path, path.split('/')[-1]))
            for path in missed_5_list:
                shutil.copyfile(path, ops.join(missed_more_path, path.split('/')[-1]))
            for path in more_list:
                shutil.copyfile(path,ops.join(pred_more_path,path.split('/')[-1]))

    return


if __name__ == '__main__':

    generate_prediction_result('/home/stevemaary/data/', '/home/stevemaary/data/pred',
                               './model/tusimple_lanenet/tusimple_lanenet_vgg.ckpt')

    import sys
    try:
        # if len(sys.argv) != 3:
        #    raise Exception('Invalid input arguments')
        print(LaneEval.bench_one_submit("/home/stevemaary/data/pred/result.json", "/home/stevemaary/data/test_label.json"))
    except Exception as e:
        print(e.message)
        sys.exit(e.message)

    compare_pred_label("/home/stevemaary/data/pred/result.json", "/home/stevemaary/data/test_label.json", "/home/stevemaary/data/pred/missed_lanes.json", False)
    copy_missed_results("/home/stevemaary/data/pred/missed_lanes.json", "/home/stevemaary/data/pred/missed/")
    compare_pred_label("/home/stevemaary/data/pred/result.json", "/home/stevemaary/data/test_label.json", "/home/stevemaary/data/pred/missed_lanes.json", True)
