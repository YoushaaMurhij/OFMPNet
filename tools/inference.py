import os
import zlib
import glob
import torch
import numpy as np
from tqdm import tqdm

from google.protobuf import text_format
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from waymo_open_dataset.protos import occupancy_flow_submission_pb2

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from core.models.OFMPNet import OFMPNet
from core.datasets.WODataset import WODataset
import core.utils.occupancy_flow_grids as occupancy_flow_grids


#configuration
config = occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig()
with open('configs/waymo_ofp.config', 'r') as f:
    config_text = f.read()
    text_format.Parse(config_text, config)
print(config)

# Hyper parameters
NUM_PRED_CHANNELS = 4

def parse_record_test(features):

    features['centerlines'] = features['centerlines'].to(torch.float32)
    features['actors'] = features['actors'].to(torch.float32)
    features['occl_actors'] = features['occl_actors'].to(torch.float32)
    features['ogm'] = features['ogm'].to(torch.float32)
    features['map_image'] = (features['map_image'].to(torch.float32) / 256)
    features['vec_flow'] = features['vec_flow']
    features['scenario/id'] = features['scenario/id']

    return features


def _get_pred_waypoint_logits(
    model_outputs: torch.Tensor,
    mode_flow_outputs:torch.Tensor=None) -> occupancy_flow_grids.WaypointGrids:
  """Slices model predictions into occupancy and flow grids."""
  pred_waypoint_logits = occupancy_flow_grids.WaypointGrids()

  # Slice channels into output predictions.
  for k in range(config.num_waypoints):
    index = k * NUM_PRED_CHANNELS
    if mode_flow_outputs is not None:
        waypoint_channels_flow = mode_flow_outputs[:, :, :, index:index + NUM_PRED_CHANNELS]
    waypoint_channels = model_outputs[:, :, :, index:index + NUM_PRED_CHANNELS]
    pred_observed_occupancy = waypoint_channels[:, :, :, :1]
    pred_occluded_occupancy = waypoint_channels[:, :, :, 1:2]
    pred_flow = waypoint_channels[:, :, :, 2:]
    if mode_flow_outputs is not None:
        pred_flow = waypoint_channels_flow[:, :, :, 2:]
    pred_waypoint_logits.vehicles.observed_occupancy.append(
        pred_observed_occupancy)
    pred_waypoint_logits.vehicles.occluded_occupancy.append(
        pred_occluded_occupancy)
    pred_waypoint_logits.vehicles.flow.append(pred_flow)

  return pred_waypoint_logits

def _apply_sigmoid_to_occupancy_logits(
    pred_waypoint_logits: occupancy_flow_grids.WaypointGrids
) -> occupancy_flow_grids.WaypointGrids:
  """Converts occupancy logits with probabilities."""
  pred_waypoints = occupancy_flow_grids.WaypointGrids()
  pred_waypoints.vehicles.observed_occupancy = [
      torch.sigmoid(x) for x in pred_waypoint_logits.vehicles.observed_occupancy
  ]
  pred_waypoints.vehicles.occluded_occupancy = [
      torch.sigmoid(x) for x in pred_waypoint_logits.vehicles.occluded_occupancy
  ]
  pred_waypoints.vehicles.flow = pred_waypoint_logits.vehicles.flow
  return pred_waypoints



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('load_model...')

cfg=dict(input_size=(512,512), window_size=8, embed_dim=96, depths=[2,2,2], num_heads=[3,6,12])
model = OFMPNet(cfg,actor_only=True,sep_actors=False,fg_msa=True, fg=True)
model.to(device)


os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"
init_process_group(backend="nccl", rank=0, world_size=1)
model = DDP(model, device_ids=[device])


def test_step(data):
    map_img = data['map_image'].to(device)
    centerlines = data['centerlines'].to(device)
    actors = data['actors'].to(device)
    occl_actors = data['occl_actors'].to(device)
    ogm = data['ogm'].to(device)
    flow = data['vec_flow'].to(device)

    outputs = model(ogm,map_img,obs=actors,occ=occl_actors,mapt=centerlines,flow=flow)
    logits = _get_pred_waypoint_logits(outputs)
    
    pred_waypoints = _apply_sigmoid_to_occupancy_logits(logits)

    return pred_waypoints

def _add_waypoints_to_scenario_prediction(
    pred_waypoints: occupancy_flow_grids.WaypointGrids,
    scenario_prediction: occupancy_flow_submission_pb2.ScenarioPrediction,
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
) -> None:
  """Add predictions for all waypoints to scenario_prediction message."""
  for k in range(config.num_waypoints):
    waypoint_message = scenario_prediction.waypoints.add()
    # Observed occupancy.
    obs_occupancy = pred_waypoints.vehicles.observed_occupancy[k].cpu().detach().numpy()
    obs_occupancy_quantized = np.round(obs_occupancy * 255).astype(np.uint8)
    obs_occupancy_bytes = zlib.compress(obs_occupancy_quantized.tobytes())
    waypoint_message.observed_vehicles_occupancy = obs_occupancy_bytes
    # Occluded occupancy.
    occ_occupancy = pred_waypoints.vehicles.occluded_occupancy[k].cpu().detach().numpy()
    occ_occupancy_quantized = np.round(occ_occupancy * 255).astype(np.uint8)
    occ_occupancy_bytes = zlib.compress(occ_occupancy_quantized.tobytes())
    waypoint_message.occluded_vehicles_occupancy = occ_occupancy_bytes
    # Flow.
    flow = pred_waypoints.vehicles.flow[k].cpu().detach().numpy()
    flow_quantized = np.clip(np.round(flow), -128, 127).astype(np.int8)
    flow_bytes = zlib.compress(flow_quantized.tobytes())
    waypoint_message.all_vehicles_flow = flow_bytes


def model_testing(test_path, shard, ids):
    print(f'Creating submission for test shard {shard}...')
    test_loader = _make_test_loader(test_path=test_path, shard=shard)
    submission = _make_submission_proto()

    cnt_sample = 0
    for batch in tqdm(test_loader):
        pred_waypoints = test_step(batch)

        scenario_prediction = submission.scenario_predictions.add()
        sc_id = batch['scenario/id'][0]
        if isinstance(sc_id, bytes):
            sc_id=str(sc_id, encoding = "utf-8") 
        scenario_prediction.scenario_id = sc_id

        assert sc_id in ids, (sc_id)

        # Add all waypoints.
        _add_waypoints_to_scenario_prediction(
            pred_waypoints=pred_waypoints,
            scenario_prediction=scenario_prediction,
            config=config)

        cnt_sample += 1
        
    _save_submission_to_file(submission, shard)

    return cnt_sample
        
def _make_submission_proto(
) -> occupancy_flow_submission_pb2.ChallengeSubmission:
    """Makes a submission proto to store predictions for one shard."""
    submission = occupancy_flow_submission_pb2.ChallengeSubmission()
    submission.account_name = args.account
    submission.unique_method_name = args.method
    submission.authors.extend([args.author.replace('_', ' ')])
    submission.description = args.description
    submission.method_link = ''
    return submission

def _save_submission_to_file(
    submission: occupancy_flow_submission_pb2.ChallengeSubmission,
    shard: str,
) -> None:
    """Save predictions for one test shard as a binary protobuf."""

    save_folder = args.save_dir
    os.makedirs(save_folder, exist_ok=True)
    submission_basename = 'occupancy_flow_submission.binproto' + '-' + shard + '-of-00150'

    submission_shard_file_path = os.path.join(save_folder, submission_basename)
    num_scenario_predictions = len(submission.scenario_predictions)
    print(f'Saving {num_scenario_predictions} scenario predictions to '
        f'{submission_shard_file_path}...\n')
    f = open(submission_shard_file_path, 'wb')
    f.write(submission.SerializeToString())
    f.close()

def _make_test_loader(test_path: str, shard: str) -> torch.utils.data.Dataset:
    """Makes a dataloader for one shard in the test set."""
    shard_files = glob.glob(test_path + f'/{shard}*.npz')
    test_dataset = WODataset(
        files=shard_files, 
        transform=parse_record_test
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=1, 
        pin_memory=True,
        num_workers=2,
    )
    return test_loader

def id_checking(test=True):
    if test:
        path = f'{args.ids_dir}/testing_scenario_ids.txt'
    else:
        path = f'{args.ids_dir}/validation_scenario_ids.txt'

    with open(path, 'r') as f:
        test_scenario_ids = f.readlines()
        test_scenario_ids = [id.rstrip() for id in test_scenario_ids]
        print(f'original ids num:{len(test_scenario_ids)}')
        test_scenario_ids = set(test_scenario_ids)
    return test_scenario_ids


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--ids_dir', type=str, help='ids.txt downloads from Waymos', default="./Waymo_Dataset/occupancy_flow_challenge")
    parser.add_argument('--save_dir', type=str, help='saving directory',default="./Waymo_Dataset/inference/torch")
    parser.add_argument('--file_dir', type=str, help='Test Dataset directory',default="./Waymo_Dataset/preprocessed_data/")
    parser.add_argument('--weight_path', type=str, help='Model weights directory',default="./experiments/model_1.pt")
    parser.add_argument('--split', type=str, help='Test or Val split', default="Val")
    parser.add_argument('--method', type=str, help='Unique method name', default="OFMPNet")
    parser.add_argument('--description', type=str, help='Unique method name', default="Multi-Model Transformer with LSTM")
    parser.add_argument('--account', type=str, help='Account email', default="")
    parser.add_argument('--author', type=str, help='Author name', default="")
    args = parser.parse_args()

    if args.split == 'Val':
        TEST = False
        args.save_dir += '/' + args.split
        args.file_dir += 'val_numpy'
        v_filenames = glob.glob(args.file_dir + "/*.npz")
        print(f'{len(v_filenames)} val files found, start loading dataset')
    else:
        TEST = True
        args.save_dir += '/' + args.split
        args.file_dir += 'test_numpy'
        v_filenames = glob.glob(args.file_dir + "/*.npz")
        print(f'{len(v_filenames)} test files found, start loading dataset')    
        
    checkpoint = torch.load(args.weight_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
 
    shards = set(map(lambda f: f.split('/')[-1].split('_')[0], v_filenames))
    print(f'{len(shards)} found, start loading dataset')
    test_scenario_ids = id_checking(test=TEST)
    cnt = 0
    for shard in shards:
        num = model_testing(test_path=args.file_dir, shard=shard,ids=test_scenario_ids)
        cnt += num
    print(cnt)

    destroy_process_group()

