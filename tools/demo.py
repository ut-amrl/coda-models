import argparse
import glob
from pathlib import Path
import time
import copy
import json
import os

import numpy as np
import torch


from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger)
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError
        input_dict = {
            'points': points,
            'frame_id': index,
        }
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='my_things/32lidar/pvrcnn_allclass32oracle_coda.yaml', help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='my_things/32lidar/velodyne/', help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default='my_things/32lidar/checkpoint_epoch_50.pth', help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False, root_path=Path(args.data_path), ext=args.ext, logger=logger)
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)  # TODO: why cpu?
    model.cuda()
    model.eval()

    class_names = ['Car', 'Pedestrian', 'Cyclist', 'PickupTruck', 'DeliveryTruck', 'ServiceVehicle', 'UtilityVehicle', 'Scooter', 'Motorcycle', 'FireHydrant', 'FireAlarm', 'ParkingKiosk', 'Mailbox', 'NewspaperDispenser', 'SanitizerDispenser',
                   'CondimentDispenser', 'ATM', 'VendingMachine', 'DoorSwitch', 'EmergencyAidKit', 'Computer', 'Television', 'Dumpster', 'TrashCan', 'VacuumCleaner', 'Cart', 'Chair', 'Couch', 'Bench', 'Table', 'Bollard', 'ConstructionBarrier', 'Fence', 'Railing', 'Cone', 'Stanchion', 'TrafficLight', 'TrafficSign', 'TrafficArm', 'Canopy', 'BikeRack', 'Pole', 'InformationalSign', 'WallSign', 'Door', 'FloorSign', 'RoomLabel', 'FreestandingPlant', 'Tree', 'Other']

    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):  # Looping over all timesteps
            # Now we are analysing a single timestep's point cloud
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)

            pred_dicts, _ = model.forward(data_dict)

            # x y z l w h yaw
            pc_filepath = demo_dataset.sample_file_list[idx]
            frame = pc_filepath.split('/')[-1].split('.')[0]
            traj_idx = int(frame[:2])
            frame_idx = int(frame[2:])
            pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()
            json_dict = {
                "3dbbox": [],
                "filetype": "json",
                "frame": str(frame_idx)
            }
            obj_dict_template = {
                "classId": "",
                "instanceId": "",
                "cX": 0.0,
                "cY": 0.0,
                "cZ": 0.0,
                "h": 0.0,
                "l": 0.0,
                "w": 0.0,
                "labelAttributes": {
                    "isOccluded": "None"
                },
                "r": 0.0,
                "p": 0.0,
                "y": 0.0
            }

            for box_idx, box in enumerate(pred_boxes):  # loop over all predicted boxes
                label_idx = pred_labels[box_idx]
                label_name = class_names[label_idx - 1]
                obj_dict = copy.deepcopy(obj_dict_template)
                obj_dict["cX"] = float(box[0])
                obj_dict["cY"] = float(box[1])
                obj_dict["cZ"] = float(box[2])
                obj_dict["l"] = float(box[3])
                obj_dict["w"] = float(box[4])
                obj_dict["h"] = float(box[5])
                obj_dict["y"] = float(box[6])
                obj_dict["classId"] = label_name
                obj_dict["instanceId"] = "%s:%i" % (label_name, box_idx)
                json_dict["3dbbox"].append(obj_dict)

            json_filename = "3d_bbox_os1_%i_%i.json" % (traj_idx, frame_idx)
            traj_dir = os.path.join("preds", str(traj_idx))
            if not os.path.exists(traj_dir):
                os.makedirs(traj_dir)
            json_path = os.path.join(traj_dir, json_filename)
            json_file = open(json_path, "w+")
            json_file.write(json.dumps(json_dict, indent=2))
            json_file.close()

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
