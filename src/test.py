import argparse
# import cv2  # hint: use CV2 for reading videos
from pathlib import Path
import torch

import my_transforms


def test():
    print('TODO: do detection loop on a video or a set of frames')


if __name__ == '__main__':
    # E.g. data/match1
    parser = argparse.ArgumentParser(description='Dataset generator and dataloader.')
    parser.add_argument('video_path', type=str, help='Video path.')
    parser.add_argument('model_path', type=str, default='models', help='Model path.')

    args = parser.parse_args()

    split_path = Path(args.video_path)

    print('TODO: Import model, define distance and optimizer')
    my_model = 'MyModel()'
    distance = 'MyDistance()'
    optimizer = 'My optimizer'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using: ' + str(device))
    my_model = my_model.to(device)

    model_path = Path(args.model_path)
    my_model_name = 'my_model'
    model_path = model_path / my_model_name

    print('TODO: load model')

    test()
