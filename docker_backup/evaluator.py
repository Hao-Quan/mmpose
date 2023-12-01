# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Iterator, List, Optional, Sequence, Union

from mmengine.dataset import pseudo_collate
from mmengine.registry import EVALUATOR, METRICS
from mmengine.structures import BaseDataElement
from .metric import BaseMetric

import cv2
import numpy as np

@EVALUATOR.register_module()
class Evaluator:
    """Wrapper class to compose multiple :class:`BaseMetric` instances.

    Args:
        metrics (dict or BaseMetric or Sequence): The config of metrics.
    """

    def __init__(self, metrics: Union[dict, BaseMetric, Sequence]):
        self._dataset_meta: Optional[dict] = None
        if not isinstance(metrics, Sequence):
            metrics = [metrics]
        self.metrics: List[BaseMetric] = []
        for metric in metrics:
            if isinstance(metric, dict):
                self.metrics.append(METRICS.build(metric))
            else:
                self.metrics.append(metric)

    @property
    def dataset_meta(self) -> Optional[dict]:
        """Optional[dict]: Meta info of the dataset."""
        return self._dataset_meta

    @dataset_meta.setter
    def dataset_meta(self, dataset_meta: dict) -> None:
        """Set the dataset meta info to the evaluator and it's metrics."""
        self._dataset_meta = dataset_meta
        for metric in self.metrics:
            metric.dataset_meta = dataset_meta

    def process(self,
                data_samples: Sequence[BaseDataElement],
                data_batch: Optional[Any] = None):
        """Convert ``BaseDataSample`` to dict and invoke process method of each
        metric.

        Args:
            data_samples (Sequence[BaseDataElement]): predictions of the model,
                and the ground truth of the validation set.
            data_batch (Any, optional): A batch of data from the dataloader.
        """

        '''plot for debugging purpose'''
        color = (0, 255, 0)
        markerType = cv2.MARKER_DIAMOND
        markerSize = 15
        thickness = 2

        for data_sample in data_samples:
            img_debug = cv2.imread(data_sample.img_path)

            # remove 1 head
            data_sample.pred_instances.keypoints = np.delete(data_sample.pred_instances.keypoints, 0, axis=1)
            data_sample.pred_instances.keypoint_scores = np.delete(data_sample.pred_instances.keypoint_scores, 0, axis=1)
            # remove 4 right ear
            data_sample.pred_instances.keypoints = np.delete(data_sample.pred_instances.keypoints, 2, axis=1)
            data_sample.pred_instances.keypoint_scores = np.delete(data_sample.pred_instances.keypoint_scores, 2,
                                                                   axis=1)
            # remove 5 right ear
            data_sample.pred_instances.keypoints = np.delete(data_sample.pred_instances.keypoints, 2, axis=1)
            data_sample.pred_instances.keypoint_scores = np.delete(data_sample.pred_instances.keypoint_scores, 2,
                                                                   axis=1)
            
            # add 3 dump joints
            data_sample.pred_instances.keypoints = np.concatenate((data_sample.pred_instances.keypoints, np.array([[[0, 0]]])), axis=1)
            data_sample.pred_instances.keypoints = np.concatenate(
                (data_sample.pred_instances.keypoints, np.array([[[0, 0]]])), axis=1)
            data_sample.pred_instances.keypoints = np.concatenate(
                (data_sample.pred_instances.keypoints, np.array([[[0, 0]]])), axis=1)
            # add 3 dump joints' scores
            data_sample.pred_instances.keypoint_scores = np.concatenate(
                (data_sample.pred_instances.keypoint_scores, np.array([[0]])), axis=1)
            data_sample.pred_instances.keypoint_scores = np.concatenate(
                (data_sample.pred_instances.keypoint_scores, np.array([[0]])), axis=1)
            data_sample.pred_instances.keypoint_scores = np.concatenate(
                (data_sample.pred_instances.keypoint_scores, np.array([[0]])), axis=1)

            '''start plotting for debugging purpose'''
            # for i in range(17):
            # # for i in range(14):
            #     cv2.circle(img_debug, (int(data_sample.pred_instances.keypoints[0][i][0]), int(data_sample.pred_instances.keypoints[0][i][1])), radius=2, color=(0, 0, 255), thickness=2)
            #     font = cv2.FONT_HERSHEY_SIMPLEX
            #     org = (int(data_sample.pred_instances.keypoints[0][i][0]) + 70, int(data_sample.pred_instances.keypoints[0][i][1]))
            #     fontScale = 0.4
            #     # RED number
            #     fontcolor = (0, 0, 255)
            #     cv2.putText(img_debug, str(i+1), org, font, fontScale, fontcolor, thickness, cv2.LINE_AA)
            #
            # # plot orig JRDB annotations
            # # for i in range(17):
            # for i in range(17):
            #     cv2.drawMarker(img_debug, (int(data_sample.raw_ann_info["keypoints"][i*3]),
            #                                          int(data_sample.raw_ann_info["keypoints"][i*3+1])), color, markerType, markerSize, thickness)
            #     font = cv2.FONT_HERSHEY_SIMPLEX
            #     org = (int(data_sample.raw_ann_info["keypoints"][i*3]) - 70, int(data_sample.raw_ann_info["keypoints"][i*3+1]))
            #     fontScale = 0.4
            #     # GREEN number
            #     fontcolor = (0, 255, 0)
            #     cv2.putText(img_debug, str(i+1), org, font, fontScale, fontcolor, thickness, cv2.LINE_AA)
            #
            # cv2.imshow("res", img_debug)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        
        '''end plotting for debugging purpose'''
                
        _data_samples = []
        for data_sample in data_samples:
            if isinstance(data_sample, BaseDataElement):
                _data_samples.append(data_sample.to_dict())
            else:
                _data_samples.append(data_sample)

        for metric in self.metrics:
            metric.process(data_batch, _data_samples)

    def evaluate(self, size: int) -> dict:
        """Invoke ``evaluate`` method of each metric and collect the metrics
        dictionary.

        Args:
            size (int): Length of the entire validation dataset. When batch
                size > 1, the dataloader may pad some data samples to make
                sure all ranks have the same length of dataset slice. The
                ``collect_results`` function will drop the padded data based on
                this size.

        Returns:
            dict: Evaluation results of all metrics. The keys are the names
            of the metrics, and the values are corresponding results.
        """
        metrics = {}
        for metric in self.metrics:
            _results = metric.evaluate(size)

            # Check metric name conflicts
            for name in _results.keys():
                if name in metrics:
                    raise ValueError(
                        'There are multiple evaluation results with the same '
                        f'metric name {name}. Please make sure all metrics '
                        'have different prefixes.')

            metrics.update(_results)
        return metrics

    def offline_evaluate(self,
                         data_samples: Sequence,
                         data: Optional[Sequence] = None,
                         chunk_size: int = 1):
        """Offline evaluate the dumped predictions on the given data .

        Args:
            data_samples (Sequence): All predictions and ground truth of the
                model and the validation set.
            data (Sequence, optional): All data of the validation set.
            chunk_size (int): The number of data samples and predictions to be
                processed in a batch.
        """

        # support chunking iterable objects
        def get_chunks(seq: Iterator, chunk_size=1):
            stop = False
            while not stop:
                chunk = []
                for _ in range(chunk_size):
                    try:
                        chunk.append(next(seq))
                    except StopIteration:
                        stop = True
                        break
                if chunk:
                    yield chunk

        if data is not None:
            assert len(data_samples) == len(data), (
                'data_samples and data should have the same length, but got '
                f'data_samples length: {len(data_samples)} '
                f'data length: {len(data)}')
            data = get_chunks(iter(data), chunk_size)

        size = 0
        for output_chunk in get_chunks(iter(data_samples), chunk_size):
            if data is not None:
                data_chunk = pseudo_collate(next(data))  # type: ignore
            else:
                data_chunk = None
            size += len(output_chunk)
            self.process(output_chunk, data_chunk)
        return self.evaluate(size)
