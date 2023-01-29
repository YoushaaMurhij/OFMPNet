import dataclasses
from typing import List, Mapping, Optional
import torch

import core.utils.occupancy_flow_data as occupancy_flow_data

from waymo_open_dataset.protos import scenario_pb2

_ObjectType = scenario_pb2.Track.ObjectType



# Holds num_waypoints occupancy and flow tensors for one agent class.
@dataclasses.dataclass
class _WaypointGridsOneType:
  """Sequence of num_waypoints occupancy and flow tensors for one agent type."""
  # num_waypoints tensors shaped [batch_size, height, width, 1]
  observed_occupancy: List[torch.Tensor] = dataclasses.field(default_factory=list)
  # num_waypoints tensors shaped [batch_size, height, width, 1]
  occluded_occupancy: List[torch.Tensor] = dataclasses.field(default_factory=list)
  # num_waypoints tensors shaped [batch_size, height, width, 2]
  flow: List[torch.Tensor] = dataclasses.field(default_factory=list)
  # The origin occupancy for each flow waypoint.  Notice that a flow field
  # transforms some origin occupancy into some destination occupancy.
  # Flow-origin occupancies are the base occupancies for each flow field.
  # num_waypoints tensors shaped [batch_size, height, width, 1]
  flow_origin_occupancy: List[torch.Tensor] = dataclasses.field(
      default_factory=list)


# Holds num_waypoints occupancy and flow tensors for all agent clases.  This is
# used to store both ground-truth and predicted topdowns.
@dataclasses.dataclass
class WaypointGrids:
  """Occupancy and flow sequences for vehicles, pedestrians, cyclists."""
  vehicles: _WaypointGridsOneType = dataclasses.field(
      default_factory=_WaypointGridsOneType)
  pedestrians: _WaypointGridsOneType = dataclasses.field(
      default_factory=_WaypointGridsOneType)
  cyclists: _WaypointGridsOneType = dataclasses.field(
      default_factory=_WaypointGridsOneType)

  def view(self, agent_type: str) -> _WaypointGridsOneType:
    """Retrieve occupancy and flow sequences for given agent type."""
    if agent_type == _ObjectType.TYPE_VEHICLE:
      return self.vehicles
    elif agent_type == _ObjectType.TYPE_PEDESTRIAN:
      return self.pedestrians
    elif agent_type == _ObjectType.TYPE_CYCLIST:
      return self.cyclists
    else:
      raise ValueError(f'Unknown agent type:{agent_type}.')

  def get_observed_occupancy_at_waypoint(
      self, k: int) -> occupancy_flow_data.AgentGrids:
    """Returns occupancies of currently-observed agents at waypoint k."""
    agent_grids = occupancy_flow_data.AgentGrids()
    if self.vehicles.observed_occupancy:
      agent_grids.vehicles = self.vehicles.observed_occupancy[k]
    if self.pedestrians.observed_occupancy:
      agent_grids.pedestrians = self.pedestrians.observed_occupancy[k]
    if self.cyclists.observed_occupancy:
      agent_grids.cyclists = self.cyclists.observed_occupancy[k]
    return agent_grids

  def get_occluded_occupancy_at_waypoint(
      self, k: int) -> occupancy_flow_data.AgentGrids:
    """Returns occupancies of currently-occluded agents at waypoint k."""
    agent_grids = occupancy_flow_data.AgentGrids()
    if self.vehicles.occluded_occupancy:
      agent_grids.vehicles = self.vehicles.occluded_occupancy[k]
    if self.pedestrians.occluded_occupancy:
      agent_grids.pedestrians = self.pedestrians.occluded_occupancy[k]
    if self.cyclists.occluded_occupancy:
      agent_grids.cyclists = self.cyclists.occluded_occupancy[k]
    return agent_grids

  def get_flow_at_waypoint(self, k: int) -> occupancy_flow_data.AgentGrids:
    """Returns flow fields of all agents at waypoint k."""
    agent_grids = occupancy_flow_data.AgentGrids()
    if self.vehicles.flow:
      agent_grids.vehicles = self.vehicles.flow[k]
    if self.pedestrians.flow:
      agent_grids.pedestrians = self.pedestrians.flow[k]
    if self.cyclists.flow:
      agent_grids.cyclists = self.cyclists.flow[k]
    return agent_grids