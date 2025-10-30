import itertools
import logging
import inspect
from collections import OrderedDict

import numpy as np
from mmcv.utils import print_log
try:
    from aitodpycocotools.cocoeval import COCOeval  # type: ignore
except ImportError:
    from pycocotools.cocoeval import COCOeval  # fallback when aitod tools unavailable
from terminaltables import AsciiTable

from .builder import DATASETS
from .coco import CocoDataset
from .custom import CustomDataset


@DATASETS.register_module()
class AITODDataset(CocoDataset):

    CLASSES = ('airplane', 'bridge', 'storage-tank', 'ship', 'swimming-pool', 'vehicle', 'person', 'wind-mill')

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=True,
                 classwise_lrp=True,
                 proposal_nums=(100, 300, 1500),
                 iou_thrs=None,
                 metric_items=None,
                 with_lrp=True):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        def _accumulate_with_optional_lrp(coco_eval, request_lrp):
            """Run accumulate while gracefully handling missing LRP support."""
            try:
                params = inspect.signature(coco_eval.accumulate).parameters
                supports_lrp = 'with_lrp' in params
            except (ValueError, TypeError):
                supports_lrp = False

            if supports_lrp:
                coco_eval.accumulate(with_lrp=request_lrp)
                return bool(request_lrp)

            if request_lrp:
                print_log(
                    ('COCOeval.accumulate() does not accept "with_lrp"; '
                     'continuing without LRP metrics.'),
                    logger=logger,
                    level=logging.WARNING)
            coco_eval.accumulate()
            return False

        custom_metric_map = {
            'mAP': 0,
            'mAP_25': 1,
            'mAP_50': 2,
            'mAP_75': 3,
            'mAP_vt': 4,
            'mAP_t': 5,
            'mAP_s': 6,
            'mAP_m': 7,
            'mAP_l': None,
            'AR@100': 8,
            'AR@300': 9,
            'AR@1000': None,
            'AR@1500': 10,
            'AR_vt@1500': 11,
            'AR_t@1500': 12,
            'AR_s@1000': None,
            'AR_s@1500': 13,
            'AR_m@1000': None,
            'AR_m@1500': 14,
            'AR_l@1000': None,
            'AR_l@1500': None,
            'oLRP': 15,
            'oLRP_Localisation': 16,
            'oLRP_false_positive': 17,
            'oLRP_false_negative': 18
        }
        basic_metric_map = {
            'mAP': 0,
            'mAP_50': 1,
            'mAP_75': 2,
            'mAP_s': 3,
            'mAP_m': 4,
            'mAP_l': 5,
            'mAP_25': None,
            'mAP_vt': None,
            'mAP_t': None,
            'AR@1': 6,
            'AR@10': 7,
            'AR@100': 8,
            'AR@300': None,
            'AR@1000': None,
            'AR@1500': None,
            'AR_vt@1500': None,
            'AR_t@1500': None,
            'AR_s': 9,
            'AR_s@1000': None,
            'AR_s@1500': None,
            'AR_m': 10,
            'AR_m@1000': None,
            'AR_m@1500': None,
            'AR_l': 11,
            'AR_l@1000': None,
            'AR_l@1500': None,
            'oLRP': None,
            'oLRP_Localisation': None,
            'oLRP_false_positive': None,
            'oLRP_false_negative': None
        }

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        eval_results = OrderedDict()
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)
            cur_metric_items = None if metric_items is None else list(metric_items)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                cocoDt = cocoGt.loadRes(result_files[metric])
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            iou_type = 'bbox' if metric == 'proposal' else metric
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.evaluate()
                _accumulate_with_optional_lrp(cocoEval, with_lrp)
                cocoEval.summarize()
                custom_stats = len(cocoEval.stats) >= 19
                metric_map = custom_metric_map if custom_stats else basic_metric_map
                if cur_metric_items is None:
                    cur_metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]
                filtered_items = []
                for item in cur_metric_items:
                    if item not in metric_map:
                        raise KeyError(
                            f'metric item {item} is not supported')
                    stat_idx = metric_map[item]
                    if stat_idx is None or stat_idx >= len(cocoEval.stats):
                        print_log(
                            f'Skipping metric "{item}" because it is not '
                            'available in the current COCOeval output.',
                            logger=logger,
                            level=logging.WARNING)
                        continue
                    filtered_items.append((item, stat_idx))

                for item, stat_idx in filtered_items:
                    val = float(f'{cocoEval.stats[stat_idx]:.3f}')
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                effective_with_lrp = _accumulate_with_optional_lrp(
                    cocoEval, with_lrp)
                cocoEval.summarize()
                custom_stats = len(cocoEval.stats) >= 19
                metric_map = custom_metric_map if custom_stats else basic_metric_map
                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)
                
                if classwise_lrp and effective_with_lrp and 'olrp' in cocoEval.eval:
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    oLRPs = cocoEval.eval['olrp']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == oLRPs.shape[0]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        olrp = oLRPs[idx, 0, -1]
                        olrp = olrp[olrp > -1]
                        if olrp.size:
                            ap = np.mean(olrp)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'oLRP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)
                elif classwise_lrp and with_lrp:
                    print_log(
                        'Skipping classwise LRP breakdown because COCOeval '
                        'did not report LRP metrics.',
                        logger=logger,
                        level=logging.WARNING)

                if cur_metric_items is None:
                    if custom_stats:
                        cur_metric_items = [
                            'mAP', 'mAP_50', 'mAP_75', 'mAP_vt', 'mAP_t',
                            'mAP_s', 'mAP_m'
                        ]
                        if effective_with_lrp:
                            cur_metric_items += [
                                'oLRP', 'oLRP_Localisation',
                                'oLRP_false_positive', 'oLRP_false_negative'
                            ]
                    else:
                        cur_metric_items = [
                            'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m',
                            'mAP_l'
                        ]

                filtered_metric_items = []
                for metric_item in cur_metric_items:
                    if metric_item not in metric_map:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')
                    if metric_item.startswith('oLRP') and not effective_with_lrp:
                        print_log(
                            f'Skipping metric "{metric_item}" because LRP is '
                            'not available with the current COCOeval build.',
                            logger=logger,
                            level=logging.WARNING)
                        continue
                    stat_idx = metric_map[metric_item]
                    if stat_idx is None or stat_idx >= len(cocoEval.stats):
                        print_log(
                            f'Skipping metric "{metric_item}" because '
                            'COCOeval did not report a compatible statistic.',
                            logger=logger,
                            level=logging.WARNING)
                        continue
                    filtered_metric_items.append((metric_item, stat_idx))

                for metric_item, stat_idx in filtered_metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(f'{cocoEval.stats[stat_idx]:.3f}')
                    eval_results[key] = val
                ap = cocoEval.stats[:6]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    f'{ap[4]:.3f} {ap[5]:.3f}')
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
