import os
import zlib
import glob
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm

from google.protobuf import text_format
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from waymo_open_dataset.protos import occupancy_flow_submission_pb2

import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from core.utils.visual_utils import *
from core.models.OFMPNet import OFMPNet
from core.datasets.WODataset import WODataset
import core.utils.occupancy_flow_grids as occupancy_flow_grids
import core.utils.occupancy_flow_data as occupancy_flow_data

from matplotlib.animation import PillowWriter

# configuration
config = occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig()
with open('configs/waymo_ofp.config', 'r') as f:
    config_text = f.read()
    text_format.Parse(config_text, config)
print(config)

# Hyper parameters
NUM_PRED_CHANNELS = 4
TEST = True


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
        mode_flow_outputs: torch.Tensor = None) -> occupancy_flow_grids.WaypointGrids:
    """Slices model predictions into occupancy and flow grids."""
    pred_waypoint_logits = occupancy_flow_grids.WaypointGrids()

    # Slice channels into output predictions.
    for k in range(config.num_waypoints):
        index = k * NUM_PRED_CHANNELS
        if mode_flow_outputs is not None:
            waypoint_channels_flow = mode_flow_outputs[:,
                                                       :, :, index:index + NUM_PRED_CHANNELS]
        waypoint_channels = model_outputs[:, :,
                                          :, index:index + NUM_PRED_CHANNELS]
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

cfg = dict(input_size=(512, 512), window_size=8, embed_dim=96,
           depths=[2, 2, 2], num_heads=[3, 6, 12])
model = OFMPNet(cfg, actor_only=True, sep_actors=False, fg_msa=True, fg=True)
model.to(device)


os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"
init_process_group(backend="nccl", rank=0, world_size=1)
model = DDP(model, device_ids=[device])


def infer_step(data):
    map_img = data['map_image'].to(device)
    centerlines = data['centerlines'].to(device)
    actors = data['actors'].to(device)
    occl_actors = data['occl_actors'].to(device)
    ogm = data['ogm'].to(device)
    flow = data['vec_flow'].to(device)

    torch.cuda.synchronize()
    t = time.time()
    outputs = model(ogm, map_img, obs=actors, occ=occl_actors,
                    mapt=centerlines, flow=flow)
    torch.cuda.synchronize()

    print("Inference forward time: {T:.3f}".format(T = time.time() - t))
    logits = _get_pred_waypoint_logits(outputs)

    pred_waypoints = _apply_sigmoid_to_occupancy_logits(logits)

    return pred_waypoints


def model_infer(test_path, shard, ids):
    print(f'Creating submission for test shard {shard}...')
    test_loader = _make_dataloader(test_path=test_path, shard=shard)

    cnt_sample = 0

    PAD = torch.zeros((1, 256, 3))
    PAD_H = torch.zeros((256, 1, 3))

    # ['centerlines', 'actors', 'occl_actors', 'ogm', 'map_image', 'vec_flow', 'byc_flow', 'scenario/id', 'gt_obs_ogm', 'gt_occ_ogm', 'gt_flow', 'origin_flow']
    counter = 0
    for batch in tqdm(test_loader):
        roadgraph = batch['map_image']
        pred_waypoints = infer_step(batch)

        pred_observed_occupancy_images = []
        pred_occluded_occupancy_images = []
        pred_flow_images = []

        for k in range(config.num_waypoints):
            observed_occupancy_grids = occupancy_flow_grids.WaypointGrids.get_observed_occupancy_at_waypoint(
                pred_waypoints, k)
            observed_occupancy_rgb = occupancy_rgb_image(
                agent_grids=observed_occupancy_grids,
                roadgraph_image=roadgraph,
            )
            pred_observed_occupancy_images.append(
                observed_occupancy_rgb[0].detach().cpu())

        for k in range(config.num_waypoints):
            occluded_occupancy_grids = occupancy_flow_grids.WaypointGrids.get_occluded_occupancy_at_waypoint(
                pred_waypoints, k)
            occluded_occupancy_rgb = occupancy_rgb_image(
                agent_grids=occluded_occupancy_grids,
                roadgraph_image=roadgraph,
            )
            pred_occluded_occupancy_images.append(
                occluded_occupancy_rgb[0].detach().cpu())

        for k in range(config.num_waypoints):
            flow_grids = occupancy_flow_grids.WaypointGrids.get_flow_at_waypoint(
                pred_waypoints, k)
            flow_rgb = flow_rgb_image(
                flow=flow_grids.vehicles,
                roadgraph_image=roadgraph,
                agent_trails=None,
            )
            pred_flow_images.append(flow_rgb[0].detach().cpu())

        all_images = []
        for im1, im2, im3 in zip(pred_observed_occupancy_images, pred_occluded_occupancy_images, pred_flow_images):
            all_images.append(torch.concat(
                [im1, PAD_H, im2, PAD_H, im3], axis=1))

        anim = create_animation(all_images, interval=200)
        anim.save(os.path.join(args.save_dir, 'all_' +
                  str(counter) + '.gif'), writer=PillowWriter(fps=5))

        counter += 1
        cnt_sample += 1

    return cnt_sample


def _make_dataloader(test_path: str, shard: str) -> torch.utils.data.Dataset:
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
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--ids_dir', type=str, help='ids.txt downloads from Waymos',
                        default="./Waymo_Dataset/occupancy_flow_challenge")
    parser.add_argument('--save_dir', type=str, help='saving directory',
                        default="./Waymo_Dataset/inference/visual")
    parser.add_argument('--file_dir', type=str, help='Test Dataset directory',
                        default="./Waymo_Dataset/preprocessed_data/val_numpy")
    parser.add_argument('--weight_path', type=str,
                        help='Model weights directory', default="./pretrained/epoch_15.pt")
    args = parser.parse_args()

    checkpoint = torch.load(args.weight_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    v_filenames = glob.glob(args.file_dir + "/*.npz")
    shards = set(map(lambda f: f.split('/')[-1].split('_')[0], v_filenames))
    print(f'{len(shards)} found, start loading dataset')
    test_scenario_ids = id_checking(test=TEST)
    cnt = 0
    for shard in shards:
        num = model_infer(test_path=args.file_dir,
                          shard=shard, ids=test_scenario_ids)
        cnt += num
    print(cnt)

    destroy_process_group()
