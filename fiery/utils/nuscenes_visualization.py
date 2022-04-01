
# nuScenes dev-kit.
# Code written by Holger Caesar, Varun Bankiti, and Alex Lang, 2019.

from typing import Any, List, Union

import numpy as np
from matplotlib import pyplot as plt

from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box as NuScenesBox
import torch
from fiery.utils.mm_obj_evaluation_utils import output_to_nusc_box
from fiery.utils.visualisation import heatmap_image
from pyquaternion import Quaternion
from mmdet3d.core import bbox3d2result
from mmdet3d.core.bbox import BaseInstance3DBoxes

Axis = Any

general_to_detection = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    # "human.pedestrian.wheelchair": "ignore",
    # "human.pedestrian.stroller": "ignore",
    # "human.pedestrian.personal_mobility": "ignore",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    # "animal": "ignore",
    "vehicle.car": "car",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    # "vehicle.emergency.ambulance": "ignore",
    # "vehicle.emergency.police": "ignore",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    # "movable_object.pushable_pullable": "ignore",
    # "movable_object.debris": "ignore",
    # "static_object.bicycle_rack": "ignore",
}


def visualize_sample(
    nusc: NuScenes,
    sample_token: str,
    pred_boxes: NuScenesBox,
    nsweeps: int = 1,
    conf_th: float = 0.15,
    eval_range: float = 50,
    verbose: bool = False,
) -> None:
    """
    Visualizes a sample from BEV with annotations and detection results.
    :param nusc: NuScenes object.
    :param sample_token: The nuScenes sample token.
    :param gt_boxes: Ground truth boxes grouped by sample.
    :param pred_boxes: Prediction grouped by sample.
    :param nsweeps: Number of sweeps used for lidar visualization.
    :param conf_th: The confidence threshold used to filter negatives.
    :param eval_range: Range in meters beyond which boxes are ignored.
    :param verbose: Whether to print to stdout.
    """
    #####
    # Get Sample Record
    #####
    rec = nusc.get('sample', sample_token)
    #####
    # Get GT boxes.
    #####
    gt_boxes = []
    #####
    # global coordinate to lidartop ego coordinate
    #####
    egopose = nusc.get('ego_pose',
                       nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
    ego_translation = -np.array(egopose['translation'])
    yaw = Quaternion(egopose['rotation']).yaw_pitch_roll[0]
    ego_rotation = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse

    for annotation_token in rec['anns']:
        annotation = nusc.get('sample_annotation', annotation_token)
        # Get label name in detection task and filter unused labels.
        if annotation['category_name'] not in general_to_detection:
            continue
        if int(annotation['visibility_token']) == 1:
            continue

        detection_name = general_to_detection[annotation['category_name']]

        gt_box = NuScenesBox(
            center=annotation['translation'],
            size=annotation['size'],
            orientation=Quaternion(annotation['rotation']),
            # label=labels[i],
            score=-1.0,
            # velocity=nusc.box_velocity(annotation['token'])[:2],
            name=detection_name,
            token=sample_token,
        )

        gt_box.translate(ego_translation)
        gt_box.rotate(ego_rotation)

        gt_boxes.append(gt_box)
    #####
    # Get Pred boxes.
    #####
    # Init axes.
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))

    # Show ego vehicle.
    ax.plot(0, 0, 'x', color='black')

    # Show GT boxes.
    for box in gt_boxes:
        box.render(ax, view=np.eye(4), colors=('r', 'r', 'r'), linewidth=2)

    # Show EST boxes.
    for box in pred_boxes:
        # Show only predictions with a high score.
        assert not np.isnan(box.score), 'Error: Box score cannot be NaN!'
        if box.score >= conf_th:
            box.render(ax, view=np.eye(4), colors=('g', 'g', 'g'), linewidth=1)

    # Limit visible range.
    axes_limit = eval_range + 3  # Slightly bigger to include boxes that extend beyond the range.
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)

    ax.invert_yaxis()
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Show / save plot.
    if verbose:
        print('Rendering sample token %s' % sample_token)
    plt.title(sample_token)
    return fig


def visualize_bbox(
    pred_bboxes: List[Union[BaseInstance3DBoxes, torch.Tensor]],
    gt_bbox_3d: BaseInstance3DBoxes,
    gt_label_3d: torch.Tensor,
    token: str,
    conf_th: float = 0.15,
    eval_range: float = 50,
    verbose: bool = False
) -> None:
    """Visualizes a sample from BEV with annotations and detection results.
    Args:
        pred_bboxes: Prediction grouped by sample.
        gt_bbox_3d: Ground truth boxes grouped by sample.
        gt_label_3d: Ground truch labels grouped by sample.
        token: The nuScenes sample token.
        conf_th: The confidence threshold used to filter negatives.
        eval_range: Range in meters beyond which boxes are ignored.
        verbose: Whether to print to stdout.
    """

    #####
    # Get GT boxes.
    #####
    gt_bboxes = output_to_nusc_box(bbox3d2result(gt_bbox_3d, torch.ones_like(gt_label_3d), gt_label_3d), token)

    #####
    # Get Pred boxes.
    #####
    bboxes, scores, labels = pred_bboxes
    pred_bboxes = output_to_nusc_box(bbox3d2result(bboxes, scores, labels), token)

    # Init axes.
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))

    # Show ego vehicle.
    ax.plot(0, 0, 'x', color='black')

    # Show GT boxes.
    for box in gt_bboxes:
        box.render(ax, view=np.eye(4), colors=('g', 'g', 'g'), linewidth=2)

    # Show EST boxes.
    for box in pred_bboxes:
        # Show only predictions with a high score.
        assert not np.isnan(box.score), 'Error: Box score cannot be NaN!'
        if box.score >= conf_th:
            box.render(ax, view=np.eye(4), colors=('b', 'b', 'b'), linewidth=1)

    # Limit visible range.
    axes_limit = eval_range + 3  # Slightly bigger to include boxes that extend beyond the range.
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)

    # Reverse X, Y axis
    # ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    # Show / save plot.
    if verbose:
        print('Rendering sample token %s' % token)
    plt.title(token)
    return fig


def visualize_center(pred_heatmap, gt_heatmap):
    gt_heatmap_img = heatmap_image(gt_heatmap.detach().cpu().numpy())
    pred_heatmap_img = heatmap_image(pred_heatmap.detach().cpu().numpy())
    heatmap_img = np.concatenate([gt_heatmap_img, pred_heatmap_img], axis=1)
    return heatmap_img


def visualize_depth_map(pred_depth_map, gt_depth_map):
    """Visualize depth map of prediction and ground truth
    Args:
        pred_depth_map: [num_cameras, num_classes, H, W]
        gt_depth_map: [num_cameras, H, W]

    Returns:
        The depth map of prediction and ground truth. matplotlib.pyplot.Figure
    """
    num_cameras, _, _ = gt_depth_map.shape
    f, ax = plt.subplots(2, num_cameras, figsize=(num_cameras * 2, 2))
    for i, (pred_map, gt_map) in enumerate(zip(pred_depth_map, gt_depth_map)):
        ax[0][i].imshow(gt_map.detach().cpu().numpy(), cmap='gnuplot2_r')
        ax[1][i].imshow(pred_map.detach().cpu().argmax(dim=0).numpy(), cmap='gnuplot2_r')

    f.tight_layout()
    return f
