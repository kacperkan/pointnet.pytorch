import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader, Dataset

from pointnet.model import PointNetfeat

USE_GPU = torch.cuda.is_available()


class EmbeddingDataset(Dataset):
    def __init__(self, paths: List[str]):
        self.paths = paths

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.paths[index]
        point_set = np.load(path)

        # code below directly copied from the "pointnet/dataset.py"
        point_set = point_set - np.expand_dims(
            np.mean(point_set, axis=0), 0
        )  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale
        return torch.from_numpy(point_set).float()

    def __len__(self):
        return len(self.paths)


def generate_embeddings(data_path: str, weights: str):
    data_paths = [path.as_posix() for path in Path(data_path).rglob("*.npy")]

    model = PointNetfeat(
        feature_transform=True, global_feat=True, num_out_features=32
    )
    model.load_state_dict(torch.load(weights))
    model.eval()

    if USE_GPU:
        model = model.cuda()

    dataset = EmbeddingDataset(data_paths)
    data_loader = DataLoader(
        dataset, batch_size=32, shuffle=False, drop_last=False, pin_memory=True
    )

    for batch_idx, batch in tqdm.tqdm(data_loader):  # type: int, torch.Tensor
        if USE_GPU:
            batch = batch.cuda()

        embs: torch.Tensor = model(batch)[0]
        embs = embs.detach().cpu().numpy()
        batch_paths = dataset.paths[
            batch_idx
            * data_loader.batch_size : (batch_idx + 1)
            * data_loader.batch_size
        ]

        for path, emb in zip(batch_paths, embs):
            path = Path(path[:-4]).with_suffix("_emb.npy")
            np.save(path.as_posix(), emb)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generator of embeddings from points of the PointFlow dataset of "
            "the ShapeNet"
        )
    )

    parser.add_argument(
        "--data_path", required=True, help="Path to the ShapeNetCore.v2.PC15K"
    )
    parser.add_argument(
        "--weights", required=True, help="Path to weights of the model"
    )

    args = parser.parse_args()

    generate_embeddings(**vars(args))


if __name__ == "__main__":
    main()
