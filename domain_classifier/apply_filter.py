from pathlib import Path

from nt_paths import db_root
import joblib
import numpy as np
import re
import paderbox as pb
from concurrent.futures import ProcessPoolExecutor
from distillation.data.linked_json import LinkedJsonDatabase
import torch

import click

def parse_index(str_val):
    # hacking linked json db...
    pattern = re.compile(r"@external\.(?P<file>[a-zA-Z0-9_]+)\[(?P<idx>[0-9]+)\]")
    match = pattern.match(str_val)
    if match is None:
        raise ValueError(str_val)
    file = match.group("file")
    idx = int(match.group("idx"))
    return file, idx


def get_index(ex):
    _f, idx = parse_index(ex["logits"])
    assert _f == "logits", _f
    # ex.pop("logits")
    return idx

# def domain_classifier():
#     dc_id = 1
#     model_path = (
#         f"/net/vol/werning/storage/relabel_model/domain_classifier{dc_id}/model.joblib"
#     )
#     dir_path = (
#         db_root
#         + f"/relabeling/domain_classifier{dc_id}/"
#     )


@click.command()
@click.option("--logit_database_path", default=db_root + "/relabeling/audioset_logits_full")
@click.option("--model_path", default="/net/vol/werning/storage/relabel_model/owl_peanuts_television/model.joblib")
@click.option("--output_path", default=db_root + f"/relabeling/audioset_to_target_name/")
@click.option("--num_workers", default=0)
@click.option("--keep-unlabeled-data", default=False)
def main(
    logit_database_path,
    output_path,
    model_path,
    num_workers,
    keep_unlabeled_data,
):
    db_filename = "database.json"
    out_filename = "database.json"
    output_path = Path(output_path)
    output_path_file = output_path / out_filename
    logit_database_path = Path(logit_database_path)
    logit_database_file = logit_database_path / db_filename

    assert not output_path_file.exists(), (
        f"output_path_db {output_path_file} already exists",
        output_path_file,
    )
    assert logit_database_file.exists(), (
        f"logit_database_file {logit_database_file} does not exist",
        logit_database_file,
    )
    if not output_path.exists():
        output_path.mkdir(parents=True)

    model = joblib.load(model_path)
    source_db = pb.io.load(logit_database_file)

    logits = np.load(logit_database_path / "logits.npy")

    batch_size = 10_000
    dset_len = logits.shape[0]
    print("predicting new classes...")
    if num_workers > 0:
        with ProcessPoolExecutor(num_workers) as executor:
            prediction = np.concatenate(
                list(
                    executor.map(
                        model.predict, np.array_split(logits, len(logits) // batch_size)
                    )
                )
            )
            prediction = prediction.astype(int)
    else:
        out = []
        for inputs in np.array_split(logits, len(logits) // batch_size):
            out.append(model.predict(inputs))
        prediction = np.concatenate(out)
        prediction = prediction.astype(int)
    assert prediction.shape[0] == dset_len, (
        prediction.shape,
        dset_len,
    )

    def add_example(ex_id, ex, dataset):
        index = get_index(ex)
        if prediction[index]:
            # balancing keys
            ex["event_ids"] = ex["event_ids"]
            ex["logits"] = logits[index]
            dataset[ex_id] = ex

    print("building json...")
    new_json = {}
    for dset_name, dataset in source_db["datasets"].items():
        new_dataset = {}
        for ex_id, ex in dataset.items():
            if "logits" in ex:
                add_example(ex_id, ex, new_dataset)
            # elif "segments" in ex:
            #     for seg in ex["segments"]:
            #         new_ex_id = ex_id + "_" + str(seg["start"])
            #         add_example(new_ex_id, seg, new_dataset)
            else:
                raise ValueError(ex)
        new_json[dset_name] = new_dataset
    new_db = {"datasets": new_json}
    print("done.")
    LinkedJsonDatabase.save(new_db, output_path_file)

if __name__ == "__main__":
    main()