from pathlib import Path
import argparse
import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import MyDataset
import my_transforms
# import models.MyModel --> hint, use an already existing object detector model or build your own.


def train():
    # Train params
    num_epochs = 30
    loss_to_overcome = 10000

    for epoch in range(num_epochs):
        time_now = time.time()
        for i, sample_batch in enumerate(my_dataloader):
            imgs_batch, bboxes_batch, labels_batch = sample_batch[0], sample_batch[1], sample_batch[2]
            imgs_batch = imgs_batch.to(device)
            # ===================forward=====================
            print('TODO: model forward and compute loss')
            # ===================backward====================
            print('TODO: do backpropagation of the error')

        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}, time:{}'.format(epoch + 1, num_epochs, loss.data, time.time() - time_now))

        if loss.data < loss_to_overcome:
            print('Saving model...')
            loss_to_overcome = loss.data
            torch.save(my_model.state_dict(), model_path)


if __name__ == '__main__':
    # E.g. data/match1
    parser = argparse.ArgumentParser(description='Training loop.')
    parser.add_argument('data_path', type=str, help='Data path.')
    parser.add_argument('model_path', type=str, default='models', help='Model path.')

    args = parser.parse_args()

    split_path = Path(args.data_path)

    transform = transforms.Compose([
        my_transforms.ResizeImgBbox((544, 960)),
        my_transforms.ToTensor()
    ])

    my_dataset_train = MyDataset(split_path, ['ball', 'player', 'referee', 'person'], transform=transform)
    my_dataloader = DataLoader(my_dataset_train, batch_size=32, shuffle=True, collate_fn=my_dataset_train.collate_fn,
                               num_workers=8)

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

    train()
