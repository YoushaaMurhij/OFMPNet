import torch
import torch.nn as nn
import torch.nn.functional as F
import enum
import numpy as np
import core.utils.occupancy_flow_grids as occupancy_flow_grids
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from typing import List
from torchmetrics.functional.classification import binary_average_precision

class ResamplingType(enum.Enum):
  NEAREST = 0
  BILINEAR = 1


class BorderType(enum.Enum):
  ZERO = 0
  DUPLICATE = 1


class PixelType(enum.Enum):
  INTEGER = 0
  HALF_INTEGER = 1

import enum
from typing import Optional
from typing import Union, Sequence

Integer = Union[int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16,
                np.uint32, np.uint64]
Float = Union[float, np.float16, np.float32, np.float64]
TensorLike = Union[Integer, Float, Sequence, np.ndarray, torch.Tensor]

def compute_occupancy_flow_metrics(
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
    true_waypoints: occupancy_flow_grids.WaypointGrids,
    pred_waypoints: occupancy_flow_grids.WaypointGrids,
    no_warp: bool=False
) -> occupancy_flow_metrics_pb2.OccupancyFlowMetrics:
  """Computes occupancy (observed, occluded) and flow metrics.

  Args:
    config: OccupancyFlowTaskConfig proto message.
    true_waypoints: Set of num_waypoints ground truth labels.
    pred_waypoints: Predicted set of num_waypoints occupancy and flow topdowns.

  Returns:
    OccupancyFlowMetrics proto message containing mean metric values averaged
      over all waypoints.
  """
  # Accumulate metric values for each waypoint and then compute the mean.
  metrics_dict = {
      'vehicles_observed_auc': [],
      'vehicles_occluded_auc': [],
      'vehicles_observed_iou': [],
      'vehicles_occluded_iou': [],
      'vehicles_flow_epe': [],
      'vehicles_flow_warped_occupancy_auc': [],
      'vehicles_flow_warped_occupancy_iou': [],
  }

  has_true_observed_occupancy = {-1: True}
  has_true_occluded_occupancy = {-1: True}



  # Warp flow-origin occupancies according to predicted flow fields.
  if not no_warp:
    warped_flow_origins = _flow_warp(
        config=config,
        true_waypoints=true_waypoints,
        pred_waypoints=pred_waypoints,
    )

  # Iterate over waypoints.
  for k in range(config.num_waypoints):
    true_observed_occupancy = true_waypoints.vehicles.observed_occupancy[k]
    pred_observed_occupancy = pred_waypoints.vehicles.observed_occupancy[k]
    true_occluded_occupancy = true_waypoints.vehicles.occluded_occupancy[k]
    pred_occluded_occupancy = pred_waypoints.vehicles.occluded_occupancy[k]
    true_flow = true_waypoints.vehicles.flow[k]
    pred_flow = pred_waypoints.vehicles.flow[k]

    # adding this CAUSE DISTRIBUTE ERROR!!!!
    # has_true_observed_occupancy[k] = tf.reduce_max(true_observed_occupancy) > 0
    # has_true_occluded_occupancy[k] = tf.reduce_max(true_occluded_occupancy) > 0
    # has_true_flow = (has_true_observed_occupancy[k] and
    #                   has_true_observed_occupancy[k - 1]) or (
    #                       has_true_occluded_occupancy[k] and
    #                       has_true_occluded_occupancy[k - 1])

    # Compute occupancy metrics.
    if True:#:has_true_observed_occupancy[k]:
      metrics_dict['vehicles_observed_auc'].append(
          _compute_occupancy_auc(true_observed_occupancy,
                                pred_observed_occupancy))
      metrics_dict['vehicles_observed_iou'].append(
        _compute_occupancy_soft_iou(true_observed_occupancy,
                                    pred_observed_occupancy))
    if True:#has_true_occluded_occupancy[k]:                       
      metrics_dict['vehicles_occluded_auc'].append(
          _compute_occupancy_auc(true_occluded_occupancy,
                                pred_occluded_occupancy))
      
      metrics_dict['vehicles_occluded_iou'].append(
          _compute_occupancy_soft_iou(true_occluded_occupancy,
                                      pred_occluded_occupancy))
      
    # Compute flow metrics.
    if True:#has_true_flow:
      metrics_dict['vehicles_flow_epe'].append(
          _compute_flow_epe(true_flow, pred_flow))

      # Compute flow-warped occupancy metrics.
      # First, construct ground-truth occupancy of all observed and occluded
      # vehicles.
      true_all_occupancy = torch.clamp(
          true_observed_occupancy + true_occluded_occupancy, 0, 1)
      # Construct predicted version of same value.
      pred_all_occupancy = torch.clamp(
          pred_observed_occupancy + pred_occluded_occupancy, 0, 1)
      # We expect to see the same results by warping the flow-origin occupancies.
      if not no_warp:
        flow_warped_origin_occupancy = warped_flow_origins[k]
        # Construct quantity that requires both prediction paths to be correct.
        flow_grounded_pred_all_occupancy = (
            pred_all_occupancy * flow_warped_origin_occupancy)
        # Now compute occupancy metrics between this quantity and ground-truth.
        # reverse the order of true and pred
        metrics_dict['vehicles_flow_warped_occupancy_auc'].append(
            _compute_occupancy_auc(true_all_occupancy, flow_grounded_pred_all_occupancy))
        metrics_dict['vehicles_flow_warped_occupancy_iou'].append(
            _compute_occupancy_soft_iou(flow_grounded_pred_all_occupancy,
                                        true_all_occupancy))

  # Compute means and return as proto message.
  metrics = occupancy_flow_metrics_pb2.OccupancyFlowMetrics()
  metrics.vehicles_observed_auc = _mean(metrics_dict['vehicles_observed_auc'])
  metrics.vehicles_occluded_auc = _mean(metrics_dict['vehicles_occluded_auc'])
  metrics.vehicles_observed_iou = _mean(metrics_dict['vehicles_observed_iou'])
  metrics.vehicles_occluded_iou = _mean(metrics_dict['vehicles_occluded_iou'])
  metrics.vehicles_flow_epe = _mean(metrics_dict['vehicles_flow_epe'])
  if not no_warp:
    metrics.vehicles_flow_warped_occupancy_auc = _mean(
        metrics_dict['vehicles_flow_warped_occupancy_auc'])
    metrics.vehicles_flow_warped_occupancy_iou = _mean(
        metrics_dict['vehicles_flow_warped_occupancy_iou'])
  return metrics #, metrics_dict

def sample(image: torch.Tensor,
           warp: torch.Tensor,
           resampling_type: ResamplingType = ResamplingType.BILINEAR,
           border_type: BorderType = BorderType.ZERO,
           pixel_type: PixelType = PixelType.HALF_INTEGER) -> torch.Tensor:
    """Samples an image at user defined coordinates.

    Note:
        The warp maps target to source. In the following, A1 to An are optional
        batch dimensions.

    Args:
        image: A tensor of shape `[B, H_i, W_i, C]`, where `B` is the batch size,
        `H_i` the height of the image, `W_i` the width of the image, and `C` the
        number of channels of the image.
        warp: A tensor of shape `[B, A_1, ..., A_n, 2]` containing the x and y
        coordinates at which sampling will be performed. The last dimension must
        be 2, representing the (x, y) coordinate where x is the index for width
        and y is the index for height.
    resampling_type: Resampling mode. Supported values are
        `ResamplingType.NEAREST` and `ResamplingType.BILINEAR`.
        border_type: Border mode. Supported values are `BorderType.ZERO` and
        `BorderType.DUPLICATE`.
        pixel_type: Pixel mode. Supported values are `PixelType.INTEGER` and
        `PixelType.HALF_INTEGER`.
        name: A name for this op. Defaults to "sample".

    Returns:
        Tensor of sampled values from `image`. The output tensor shape
        is `[B, A_1, ..., A_n, C]`.

    Raises:
        ValueError: If `image` has rank != 4. If `warp` has rank < 2 or its last
        dimension is not 2. If `image` and `warp` batch dimension does not match.
    """

    # shape.check_static(image, tensor_name="image", has_rank=4)
    # shape.check_static(
    #     warp,
    #     tensor_name="warp",
    #     has_rank_greater_than=1,
    #     has_dim_equals=(-1, 2))
    # shape.compare_batch_dimensions(
    #     tensors=(image, warp), last_axes=0, broadcast_compatible=False)

    if pixel_type == PixelType.HALF_INTEGER:
        warp -= 0.5

    if resampling_type == ResamplingType.NEAREST:
        warp = torch.math.round(warp)

    if border_type == BorderType.ZERO:
        image = F.pad(image, (0,0,1,1,1,1,0,0))
        warp = warp + 1

    warp_shape = warp.size()
    flat_warp = torch.reshape(warp, (warp_shape[0], -1, 2))
    flat_sampled = _interpolate_bilinear(image, flat_warp, indexing="xy")
    output_shape = [*warp_shape[:-1], flat_sampled.size()[-1]]
    return torch.reshape(flat_sampled, output_shape)

def _interpolate_bilinear(
    grid,
    query_points,
    indexing,
):
    """pytorch implementation of tensorflow interpolate_bilinear."""
    device = grid.device
    grid_shape = grid.size()
    query_shape = query_points.size()

    batch_size, height, width, channels = (
        grid_shape[0],
        grid_shape[1],
        grid_shape[2],
        grid_shape[3],
    )

    num_queries = query_shape[1]

    query_type = query_points.dtype
    grid_type = grid.dtype

    alphas = []
    floors = []
    ceils = []
    index_order = [0, 1] if indexing == "ij" else [1, 0]
    unstacked_query_points = torch.unbind(query_points, dim=2)

    for i, dim in enumerate(index_order):
        queries = unstacked_query_points[dim]

        size_in_indexing_dimension = grid_shape[i + 1]

        # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
        # is still a valid index into the grid.
        max_floor = torch.tensor(size_in_indexing_dimension - 2, dtype=query_type, device=device)
        min_floor = torch.tensor(0.0, dtype=query_type, device=device)
        floor = torch.minimum(
            torch.maximum(min_floor, torch.floor(queries)), max_floor
        )
        int_floor = floor.to(torch.int32)
        floors.append(int_floor)
        ceil = int_floor + 1
        ceils.append(ceil)

        # alpha has the same type as the grid, as we will directly use alpha
        # when taking linear combinations of pixel values from the image.
        alpha = (queries - floor).to(grid_type)
        min_alpha = torch.tensor(0.0, dtype=grid_type, device=device)
        max_alpha = torch.tensor(1.0, dtype=grid_type, device=device)
        alpha = torch.minimum(torch.maximum(min_alpha, alpha), max_alpha)

        # Expand alpha to [b, n, 1] so we can use broadcasting
        # (since the alpha values don't depend on the channel).
        alpha = torch.unsqueeze(alpha, 2)
        alphas.append(alpha)

        flattened_grid = torch.reshape(grid, [batch_size * height * width, channels])
        batch_offsets = torch.reshape(
            torch.arange(0, batch_size, device=device) * height * width, [batch_size, 1]
        )

    # This wraps tf.gather. We reshape the image data such that the
    # batch, y, and x coordinates are pulled into the first dimension.
    # Then we gather. Finally, we reshape the output back. It's possible this
    # code would be made simpler by using tf.gather_nd.
    def gather(y_coords, x_coords, name):
        linear_coordinates = batch_offsets + y_coords * width + x_coords
        gathered_values = flattened_grid[linear_coordinates]
        return torch.reshape(gathered_values, [batch_size, num_queries, channels])

    # grab the pixel values in the 4 corners around each query point
    top_left = gather(floors[0], floors[1], "top_left")
    top_right = gather(floors[0], ceils[1], "top_right")
    bottom_left = gather(ceils[0], floors[1], "bottom_left")
    bottom_right = gather(ceils[0], ceils[1], "bottom_right")

    # now, do the actual interpolation
    interp_top = alphas[1] * (top_right - top_left) + top_left
    interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
    interp = alphas[0] * (interp_bottom - interp_top) + interp_top

    return interp



def _mean(tensor_list: Sequence[torch.Tensor]):
  """Compute mean value from a list of scalar tensors."""
  num_tensors = len(tensor_list)
  if num_tensors == 0:
    return 0
  sum_tensors = sum(tensor_list)
  return sum_tensors / num_tensors


def _compute_occupancy_auc(
    true_occupancy: torch.Tensor,
    pred_occupancy: torch.Tensor,
) -> torch.Tensor:
  """Computes the AUC between the predicted and true occupancy grids.

  Args:
    true_occupancy: float32 [batch_size, height, width, 1] tensor in [0, 1].
    pred_occupancy: float32 [batch_size, height, width, 1] tensor in [0, 1].

  Returns:
    AUC: float32 scalar.
  """
  return binary_average_precision(preds=pred_occupancy, target=true_occupancy.to(torch.int8), thresholds=100)


def _compute_occupancy_soft_iou(
    true_occupancy: torch.Tensor,
    pred_occupancy: torch.Tensor,
) -> torch.Tensor:
  """Computes the soft IoU between the predicted and true occupancy grids.

  Args:
    true_occupancy: float32 [batch_size, height, width, 1] tensor in [0, 1].
    pred_occupancy: float32 [batch_size, height, width, 1] tensor in [0, 1].

  Returns:
    Soft IoU score: float32 scalar.
  """
  true_occupancy = torch.reshape(true_occupancy, [-1])
  pred_occupancy = torch.reshape(pred_occupancy, [-1])

  intersection = torch.mean(torch.multiply(pred_occupancy, true_occupancy))
  true_sum = torch.mean(true_occupancy)
  pred_sum = torch.mean(pred_occupancy)
  # Scenes with empty ground-truth will have a score of 0.
  score = torch.nan_to_num(torch.div(
            intersection,
            pred_sum + true_sum - intersection), posinf=0, neginf=0)
  return score


def _compute_flow_epe(
    true_flow: torch.Tensor,
    pred_flow: torch.Tensor,
) -> torch.Tensor:
  """Computes average end-point-error between predicted and true flow fields.

  Flow end-point-error measures the Euclidean distance between the predicted and
  ground-truth flow vector endpoints.

  Args:
    true_flow: float32 Tensor shaped [batch_size, height, width, 2].
    pred_flow: float32 Tensor shaped [batch_size, height, width, 2].

  Returns:
    EPE averaged over all grid cells: float32 scalar.
  """
  # [batch_size, height, width, 2]
  diff = true_flow - pred_flow
  # [batch_size, height, width, 1], [batch_size, height, width, 1]
  true_flow_dx, true_flow_dy = torch.chunk(true_flow, 2, dim=-1)
  # [batch_size, height, width, 1]
  flow_exists = torch.logical_or(
      torch.not_equal(true_flow_dx, 0.0),
      torch.not_equal(true_flow_dy, 0.0),
  )
  flow_exists = flow_exists.to(torch.float32)

  # Check shapes.
#   tf.debugging.assert_shapes([
#       (true_flow_dx, ['batch_size', 'height', 'width', 1]),
#       (true_flow_dy, ['batch_size', 'height', 'width', 1]),
#       (diff, ['batch_size', 'height', 'width', 2]),
#   ])

  diff = diff * flow_exists
  # [batch_size, height, width, 1]
  epe = torch.linalg.norm(diff, ord=2, dim=-1, keepdim=True)
  # Scalar.
  sum_epe = torch.sum(epe)
  # Scalar.
  sum_flow_exists = torch.sum(flow_exists)
  # Scalar.
  mean_epe = torch.nan_to_num(torch.div(
            sum_epe,
            sum_flow_exists), posinf=0, neginf=0)

#   tf.debugging.assert_shapes([
#       (epe, ['batch_size', 'height', 'width', 1]),
#       (sum_epe, []),
#       (mean_epe, []),
#   ])

  return mean_epe


def _flow_warp(
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
    true_waypoints: occupancy_flow_grids.WaypointGrids,
    pred_waypoints: occupancy_flow_grids.WaypointGrids,
) -> List[torch.Tensor]:
  """Warps ground-truth flow-origin occupancies according to predicted flows.

  Performs bilinear interpolation and samples from 4 pixels for each flow
  vector.

  Args:
    config: OccupancyFlowTaskConfig proto message.
    true_waypoints: Set of num_waypoints ground truth labels.
    pred_waypoints: Predicted set of num_waypoints occupancy and flow topdowns.

  Returns:
    List of `num_waypoints` occupancy grids for vehicles as float32
      [batch_size, height, width, 1] tensors.
  """

  device = pred_waypoints.vehicles.flow[0].device

  h = torch.arange(0, config.grid_height_cells, dtype=torch.float32, device=device)
  w = torch.arange(0, config.grid_width_cells, dtype=torch.float32, device=device)
  h_idx, w_idx = torch.meshgrid(h, w, indexing="xy")
  # These indices map each (x, y) location to (x, y).
  # [height, width, 2] but storing x, y coordinates.
  identity_indices = torch.stack(
      (
          w_idx.T,
          h_idx.T,
      ),
      dim=-1,
  )

  warped_flow_origins = []
  for k in range(config.num_waypoints):
    # [batch_size, height, width, 1]
    flow_origin_occupancy = true_waypoints.vehicles.flow_origin_occupancy[k]
    # [batch_size, height, width, 2]
    pred_flow = pred_waypoints.vehicles.flow[k]
    # Shifting the identity grid indices according to predicted flow tells us
    # the source (origin) grid cell for each flow vector.  We simply sample
    # occupancy values from these locations.
    # [batch_size, height, width, 2]
    warped_indices = identity_indices + pred_flow
    # Pad flow_origin with a blank (zeros) boundary so that flow vectors
    # reaching outside the grid bring in zero values instead of producing edge
    # artifacts.
    # flow_origin_occupancy = tf.pad(flow_origin_occupancy,
    #                                [[0, 0], [1, 1], [1, 1], [0, 0]])
    # Shift warped indices as well to map to the padded origin.
    # warped_indices = warped_indices + 1
    # NOTE: tensorflow graphics expects warp to contain (x, y) as well.
    # [batch_size, height, width, 2]
    warped_origin = sample(
        image=flow_origin_occupancy,
        warp=warped_indices,
        pixel_type=0,
    )
    warped_flow_origins.append(warped_origin)

  return warped_flow_origins
