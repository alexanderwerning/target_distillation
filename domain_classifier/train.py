
from pathlib import Path
from pprint import pprint

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

np.random.seed(13)

def evaluate(model, db):
    results = {}
    validate_set = db.get_validate_set()
    estimate = model.predict(validate_set['logits'])
    if estimate.ndim == 2:
        estimate = np.any(estimate, axis=-1)
    target = validate_set['target'].cpu().numpy()
    class_weight_positive = 1/np.sum(target)
    class_weight_negative = 1/(len(target)-np.sum(target))
    tp = np.logical_and(estimate, target).sum()
    fp = np.logical_and(estimate, 1-target).sum()
    tn = np.logical_and(1-estimate, 1-target).sum()
    fn = np.logical_and(1-estimate, target).sum()
    results['confusion'] = pd.DataFrame([[tp, fp],[tn,fn]],["positive", "negative"], ["true", "false"])
    tp = class_weight_positive * tp
    fp = class_weight_negative * fp
    tn = class_weight_negative * tn
    fn = class_weight_positive * fn
    results['class_weighted_confusion'] = pd.DataFrame([[tp, fp],[tn,fn]],["positive", "negative"], ["true", "false"])
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    results['weighted_precision'] = precision
    results['weighted_recall'] = recall
    return results

def get_classifier(input_size=527):
    return nn.Sequential(nn.Sigmoid(), nn.Linear(input_size, 1), nn.Sigmoid())


def train(clf, db, mixup=2.0, batch_size=600, epochs=400, lr=1e-4):
    train_losses, validate_losses = [], []
    loss_fn = nn.BCELoss()
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    clf = clf.cuda()
    validate_data = db.get_validate_set()
    # main training loop
    for e in tqdm(range(epochs)):
        for it in range(db.num_train_samples // batch_size):
            batch = db.get_train_batch(batch_size=batch_size, positive_mixup_rate=mixup)
            opt.zero_grad()
            out = clf(batch['logits'])
            loss = loss_fn(out.flatten(), batch['target'])
            loss.backward()
            opt.step()
            train_loss = loss.detach().cpu().numpy().item()
            train_losses.append(train_loss)
        # validateion
        out = clf(validate_data['logits']).flatten()
        validate_loss = loss_fn(out, validate_data['target']).detach().cpu().numpy().item()
        # print(f"Validation_loss: {validate_loss}")
        validate_losses.append(validate_loss)
    return clf, train_losses, validate_losses

from distillation.relabeling.model import NNModel
def save_model(model, model_save_path = "/net/vol/werning/storage/relabel_model/x"):
    assert isinstance(model, NNModel), type(model)
    model_save_path = Path(model_save_path)
    assert not model_save_path.exists(), f"Model save path {model_save_path} exists"

    model_save_path.mkdir(parents=True)
    joblib.dump(model, model_save_path/"model.joblib")

# clf = get_classifier(527)
#     model = NNModel(clf, threshold)
#     eval(model)