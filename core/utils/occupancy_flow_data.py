import dataclasses
from typing import Dict, Optional

import torch

from waymo_open_dataset.protos import scenario_pb2

_ObjectType = scenario_pb2.Track.ObjectType
ALL_AGENT_TYPES = [
    _ObjectType.TYPE_VEHICLE,
    _ObjectType.TYPE_PEDESTRIAN,
    _ObjectType.TYPE_CYCLIST,
]


# Holds occupancy or flow tensors for different agent classes.  This same data
# structure is used to store topdown tensors rendered from input data as well
# as topdown tensors predicted by a model.
@dataclasses.dataclass
class AgentGrids:
  """Contains any topdown render for vehicles and pedestrians."""
  vehicles: Optional[torch.Tensor] = None
  pedestrians: Optional[torch.Tensor] = None
  cyclists: Optional[torch.Tensor] = None

  def view(self, agent_type: str) -> torch.Tensor:
    """Retrieve topdown tensor for given agent type."""
    if agent_type == _ObjectType.TYPE_VEHICLE:
      return self.vehicles
    elif agent_type == _ObjectType.TYPE_PEDESTRIAN:
      return self.pedestrians
    elif agent_type == _ObjectType.TYPE_CYCLIST:
      return self.cyclists
    else:
      raise ValueError(f'Unknown agent type:{agent_type}.')
