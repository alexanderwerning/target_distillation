from pathlib import Path

import padertorch as pt
import torch
from tqdm import tqdm

from distillation.data.linked_json import LinkedJsonDatabase
from itertools import islice
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


def segment_and_index(batch, segment_length, sample_rate):
    """
    Args:
        batch: batch of examples
        segment_length: length of the segments in seconds
        sr: sample rate

        audio_data: (batch_size, 1, samples)

    >>> batch = {'audio_data': torch.arange(8*10).reshape(8, 1, 10)}
    >>> segments, num_segments = segment_and_index(batch, 4, 1)
    >>> num_segments
    2
    >>> segments.shape
    torch.Size([16, 1, 4])
    >>> segments[0, 0, :]
    tensor([0, 1, 2, 3])
    >>> segments[1, 0, :]
    tensor([4, 5, 6, 7])
    """
    segment_length = int(segment_length * sample_rate)
    audio_data = batch["audio_data"]
    truncated = audio_data[
        :, :, : audio_data.shape[-1] // segment_length * segment_length
    ]
    segments = truncated.reshape(-1, truncated.shape[1], segment_length)
    num_segments = segments.shape[0] // batch["audio_data"].shape[0]
    return segments, num_segments


@hydra.main(version_base=None, config_path="conf", config_name="ensemble_embeddings")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

# @exp.automain
# def main(dir_path, filename, batch_size, segment_length, max_batches, device=0):
    device = 0
    batch_size = 32
    max_batches = None
    dir_path = Path(cfg.dir_path)
    filename = f"database.json"
    assert not (Path(dir_path) / filename).exists(), (
        "dir_path already exists",
        Path(dir_path) / filename,
    )
    path = dir_path / filename
    if not dir_path.exists():
        dir_path.mkdir(parents=True)
    model = instantiate(cfg.model).eval().to(device)
    fe = instantiate(cfg.feature_extractor).eval().to(device)
    db = instantiate(cfg.db.loader)
    dataset_names = db.database.get_dataset_names()
    datasets = {n: db.get_dataset(n) for n in dataset_names}

    # no len for UnbatchDataset
    # print(f"Dataset lengths: {[len(ds) for ds in datasets.values()]} (* {batch_size})")
    datasets_json = {}

    for dataset_name, dataset in datasets.items():
        print(dataset_name)
        dataset_json = {}
        if max_batches is not None:
            print(f"Using only {max_batches} batches")
            dataset = islice(dataset, max_batches)

        total = max_batches #if max_batches is not None else len(dataset)

        for i, batch in tqdm(enumerate(dataset), total=total):
            with torch.cuda.amp.autocast(), torch.no_grad():
                batch = pt.data.example_to_device(batch, device)
                features = fe(batch["audio_data"].squeeze(1))
                logits, _ = model(features[:, None])
                logits_ = logits.cpu().numpy()
                for i, ex_id in enumerate(batch["example_id"]):
                    dataset_json[ex_id] = {"logits": logits_[i]}
                del logits
                del logits_
                del features
                del batch
        datasets_json[dataset_name] = dataset_json
    LinkedJsonDatabase.save({"datasets": datasets_json}, path)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()