import numpy as np
import padertorch as pt
import torch
import torch.nn as nn
from pb_sed.evaluation import instance_based
from sklearn import metrics
from torch import autocast
from torchvision.utils import make_grid
from abc import ABC

from collections import defaultdict
import numpy as np
import pytorch_lightning as L

from lightning.pytorch.utilities import grad_norm

def _to_list(scalars):
    if torch.is_tensor(scalars):
        scalars = scalars.clone().cpu().data.numpy()
    if isinstance(scalars, np.ndarray):
        scalars = scalars.flatten().tolist()
    if not isinstance(scalars, (list, tuple)):
        assert np.isscalar(scalars)
        scalars = [scalars]
    return scalars

def _detach(buffer):
    if torch.is_tensor(buffer):
        buffer = buffer.detach()
    return buffer

def empty_summary_dict():
    return dict(
        scalars=defaultdict(list),
        histograms=defaultdict(list),
        audios=dict(),
        images=dict(),
        texts=dict(),
        figures=dict(),
        timings=dict(),
        buffers=defaultdict(list),
        snapshots=dict()
    )

def update_summary(summary, review):
        allowed_keys = {
            # 'loss',  # The trainer moves the loss and losses to scalars
            # 'losses',
            'scalars',
            'histograms',
            'audios',
            'images',
            'texts',
            'figures',
            'buffers',
            'snapshots'
        }
        redundant_keys = set(review.keys()) - allowed_keys
        assert len(redundant_keys) == 0, (redundant_keys, review.keys(), allowed_keys)

        assert len(review) >= 1, review
        popped_review = {**review}  # copy for "pop"

        # note item is the pytorch function to get the value of a tensor
        for key, scalars in popped_review.pop('scalars', dict()).items():
            summary['scalars'][key].extend(_to_list(scalars))
        for key, histogram in popped_review.pop('histograms', dict()).items():
            summary['histograms'][key].extend(_to_list(histogram))
            # do not hold more than 1M values in memory
            summary['histograms'][key] = \
                summary['histograms'][key][-1000000:]
        for key, buffer in popped_review.pop('buffers', dict()).items():
            summary['buffers'][key].append(_detach(buffer))
        for key, snapshot in popped_review.pop('snapshots', dict()).items():
            summary['snapshots'][key] = _detach(snapshot)  # snapshot
        for key, audio in popped_review.pop('audios', dict()).items():
            summary['audios'][key] = audio  # snapshot
        for key, image in popped_review.pop('images', dict()).items():
            summary['images'][key] = image  # snapshot
        for key, figure in popped_review.pop('figures', dict()).items():
            summary['figures'][key] = figure  # snapshot
        for key, text in popped_review.pop('texts', dict()).items():
            assert isinstance(text, str), text
            summary['texts'][key] = text  # snapshot

        assert len(popped_review) == 0, (popped_review, review)
        return summary
    

class LightningModule(L.LightningModule):
    def __init__(self, model, optimizer, lr_schedule=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.validation_summary = empty_summary_dict()
        self.test_summary = empty_summary_dict()
    
    def training_step(self, batch, batch_idx):
        out = self.model(batch)
        batch_size = batch['audio_data'].shape[0]
        summary = self.model.review(batch, out)
        self.log("training/lr",
                 self.optimizer.param_groups[0]['lr'],
                 rank_zero_only=True,
                 batch_size=batch_size)
        for k, v in summary['scalars'].items():
            name = f"training/{k}"
            self.log(name,
                     v,
                     rank_zero_only=True,
                     batch_size=batch_size)
        return summary['loss']
    
    def validation_step(self, batch, batch_idx):
        out = self.model(batch)
        batch_size = batch['audio_data'].shape[0]
        summary = self.model.review(batch, out)
        val_loss = summary['loss']
        self.log("validation/loss",
                 val_loss,
                 sync_dist=True,
                 batch_size=batch_size)
        for k, v in summary['scalars'].items():
            name = f"validation/{k}"
            self.log(name,
                     v,
                     sync_dist=True,
                     batch_size=batch_size)
        if "loss" in summary:
            summary.pop("loss")
        update_summary(self.validation_summary, summary)

    # def test_step(self, batch, batch_idx):
    #     out = self.model(batch)
    #     batch_size = batch['audio_data'].shape[0]
    #     summary = self.model.review(batch, out)
    #     val_loss = summary['loss']
    #     self.log("test/loss",
    #              val_loss,
    #              sync_dist=True,
    #              batch_size=batch_size)
    #     for k, v in summary['scalars'].items():
    #         name = f"test/{k}"
    #         self.log(name,
    #                  v,
    #                  sync_dist=True,
    #                  batch_size=batch_size)
    #     if "loss" in summary:
    #         summary.pop("loss")
    #     update_summary(self.test_summary, summary)
    
    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        summary = _modify_summary(self.model, self.validation_summary)

        if "loss" in summary['scalars']:
            summary['scalars'].pop("loss")
        self.validation_summary = empty_summary_dict()
        for k, v in summary['scalars'].items():
            if isinstance(v, list):
                continue
            name = f"validation/{k}"
            self.log(name,
                     v,
                     sync_dist=True)
    
    # def on_test_epoch_end(self) -> None:
    #     super().on_test_epoch_end()
    #     summary = _modify_summary(self.model, self.test_summary)

    #     if "loss" in summary['scalars']:
    #         summary['scalars'].pop("loss")
    #     self.test_summary = empty_summary_dict()
    #     for k, v in summary['scalars'].items():
    #         if isinstance(v, list):
    #             continue
    #         name = f"test/{k}"
    #         self.log(name,
    #                  v,
    #                  sync_dist=True)

    
    def configure_optimizers(self):
        d = {
            "optimizer": self.optimizer,
        }
        if self.lr_schedule is not None:
            d["lr_scheduler"] = self.lr_schedule
        return d

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        total_norm = 0
        for p in self.parameters():
            if p.grad is None:
                continue
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        self.log("training/grad_norm", total_norm, rank_zero_only=True)


def _add_metrics_to_summary(self, summary, suffix):
    y = np.concatenate(summary["buffers"].pop(f"y_{suffix}"))
    if y is None:
        return

    summary["scalars"][f"num_examples_{suffix}"] = len(y)
    targets = np.concatenate(summary["buffers"].pop(f"targets_{suffix}"))

    test_labels = None

    def maybe_add_label_wise(key, values):
        if key in self.labelwise_metrics:
            for event_class, value in enumerate(values):
                if test_labels is not None:
                    event_class = test_labels[event_class]
                if self.label_mapping is not None:
                    event_class = self.label_mapping[event_class]
                summary["scalars"][f"z/{key}/{event_class}"] = value
    
    if targets.ndim == 2:
        _, f, p, r = instance_based.get_best_fscore_thresholds(targets, y)
        summary["scalars"][f"macro_fscore_{suffix}"] = f.mean()
        maybe_add_label_wise(f"fscore_{suffix}", f)

        _, er, ir, dr = instance_based.get_best_er_thresholds(targets, y)
        summary["scalars"][f"macro_error_rate_{suffix}"] = er.mean()
        maybe_add_label_wise(f"error_rate_{suffix}", er)

        lwlrap, per_class_lwlrap, weight_per_class = instance_based.lwlrap(targets, y)
        summary["scalars"][f"lwlrap_{suffix}"] = lwlrap
        maybe_add_label_wise(f"lwlrap_{suffix}", per_class_lwlrap)

        if (targets.sum(0) > 1).all():
            ap = metrics.average_precision_score(targets, y, average=None)
            summary["scalars"][f"map_{suffix}"] = np.mean(ap)
            maybe_add_label_wise(f"ap_{suffix}", ap)

            auc = metrics.roc_auc_score(targets, y, average=None)
            summary["scalars"][f"mauc_{suffix}"] = np.mean(auc)
            maybe_add_label_wise(f"auc_{suffix}", auc)

        if (targets.sum(0) > 1).all():
            top1acc = (y.argmax(-1) == targets.argmax(-1)).mean()
            summary["scalars"][f"top1acc_{suffix}"] = top1acc
    else:
        top1acc = (y.argmax(-1) == targets).mean()
        summary["scalars"][f"top1acc_{suffix}"] = top1acc
    return summary

def _modify_summary(self, summary):
    """called by the trainer before dumping a summary"""
    if f"targets_weak" in summary["buffers"]:
        # Computes fscores from scores and targets
        self.add_metrics_to_summary(summary, "weak")
    for key, image in summary["images"].items():
        # prepare image grid for tensorboard
        if image.dim() == 4 and image.shape[1] > 1:
            image = image[:, 0]
        if image.dim() == 3:
            image = image.unsqueeze(1)
        summary["images"][key] = make_grid(
            image.flip(2), normalize=True, scale_each=False, nrow=1
        )
    return summary


class Model(pt.Model):
    def __init__(
        self,
        net: nn.Module,
        feature_extractor=None,
        loss_fn=torch.nn.CrossEntropyLoss(),
    ):
        """
        Args:
            net: model
        """
        super().__init__()
        self.net = net
        self.feature_extractor = feature_extractor
        self.loss_fn = loss_fn

        self.labelwise_metrics = ()
        self.label_mapping = None

    def forward(self, batch):
        float_input = batch["audio_data"].squeeze(1)
        # self.feature_extractor.to(float_input.device)
        with autocast(
            float_input.device.type,
            enabled=False,
        ):
            mel_spec = self.feature_extractor(float_input)[:, None]
        logits, feats = self.net(mel_spec)
        return logits, feats, mel_spec

    def predict(self, batch):
        float_input = batch["audio_data"].squeeze(1)
        mel_spec = self.feature_extractor(float_input)[:, None]
        with torch.no_grad():
            logits, _ = self.net(mel_spec)
            pred = torch.softmax(logits, dim=-1)
        return pred

    def compute_loss(self, inputs, outputs):
        logits, feats, mel_spec = outputs
        targets = None
        if "weak_targets" in inputs:
            targets = inputs["weak_targets"]
        if isinstance(self.loss_fn, torch.nn.CrossEntropyLoss):
            if "target" not in inputs:
                if targets is not None and targets.dim() == 2:
                    targets = torch.argmax(targets, dim=-1).long()
                else:
                    raise ValueError("Target (index) is missing, add 'target' or 'weak_targets' keys")
            else:
                targets = inputs["target"]
        if targets is None:
            raise ValueError("Missing 'weak_targets' key")
        loss = self.loss_fn(logits, targets)

        # scalars
        summary = {
            "scalars": {
                "loss": loss,
            },
        }
        return loss, targets, summary

    def review(self, inputs, outputs):
        loss, targets, loss_summary = self.compute_loss(inputs, outputs)
        logits = outputs[0]
        mel_spec = outputs[-1]

        labeled_examples_idx = (
            ((targets < 0.00001) + (targets > 0.99999)).detach().cpu().numpy() == 1
        )
        if labeled_examples_idx.ndim == 2:
            labeled_examples_idx = labeled_examples_idx.all(-1)
        y_weak = logits.detach().cpu().numpy()[labeled_examples_idx]
        weak_targets = targets.detach().cpu().numpy()[labeled_examples_idx]
        if weak_targets.ndim == 2:
            top1acc = (y_weak.argmax(-1) == weak_targets.argmax(-1)).mean()
        else:
            top1acc = (y_weak.argmax(-1) == weak_targets).mean()
        summary = {
            "loss": loss,
            "scalars": {"top1acc_weak": top1acc},
            "images": {
                "features": mel_spec[:3].detach().cpu(),  # .numpy(),
            },
            "buffers": {
                "y_weak": y_weak,
                "targets_weak": weak_targets,
            },
        }
        for k, v in summary.items():
            if isinstance(v, dict) and k in loss_summary:
                v.update(loss_summary[k])
        return summary

    def modify_summary(self, summary):
        summary = super().modify_summary(summary)
        summary = _modify_summary(self, summary)
        return summary

    def add_metrics_to_summary(self, summary, suffix):
        _add_metrics_to_summary(self, summary, suffix)


class AbstractDistillationModel(pt.Model, ABC):

    def review(self, inputs, outputs):
        loss, targets, loss_summary = self.compute_loss(inputs, outputs)
        logits = outputs[0]
        mel_spec = outputs[-1]

        labeled_examples_idx = (
            ((targets < 0.00001) + (targets > 0.99999)).detach().cpu().numpy() == 1
        )
        if labeled_examples_idx.ndim == 2:
            labeled_examples_idx = labeled_examples_idx.all(-1)
        y_weak = logits.detach().cpu().numpy()[labeled_examples_idx]
        weak_targets = targets.detach().cpu().numpy()[labeled_examples_idx]
        if weak_targets.ndim == 2:
            top1acc = (y_weak.argmax(-1) == weak_targets.argmax(-1)).mean()
        else:
            top1acc = (y_weak.argmax(-1) == weak_targets).mean()
        summary = {
            "loss": loss,
            "scalars": {"top1acc_weak": top1acc},
            "images": {
                "features": mel_spec[:3].detach().cpu(),  # .numpy(),
            },
            "buffers": {
                "y_weak": y_weak,
                "targets_weak": weak_targets,
            },
        }
        for k, v in summary.items():
            if isinstance(v, dict) and k in loss_summary:
                v.update(loss_summary[k])
        return summary

    def modify_summary(self, summary):
        """called by the trainer before dumping a summary"""
        summary = _modify_summary(self, summary)
        summary = super().modify_summary(summary)
        return summary

    def add_metrics_to_summary(self, summary, suffix):
        _add_metrics_to_summary(self, summary, suffix)


class DistillationModel(AbstractDistillationModel):
    """Response-based distillation model.

    Loss is computed between soft score of teacher and soft score of student.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        feature_extractor=None,
        label_loss_prop=0.1,
        kd_loss=torch.nn.BCEWithLogitsLoss(),
        label_loss=torch.nn.BCEWithLogitsLoss(),
    ):
        """
        Args:
            teacher: teacher model
        """
        super().__init__()
        self.teacher = teacher
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.student = student
        self.feature_extractor = feature_extractor
        self.label_loss_prop = label_loss_prop
        self.kd_loss = kd_loss
        self.label_loss = label_loss

        self.labelwise_metrics = ()
        self.label_mapping = None

        print(
            f"Student Parameters: {sum([p.numel() for p in self.student.parameters()])}"
        )
        print(
            f"Teacher Parameters: {sum([p.numel() for p in self.teacher.parameters()])}"
        )

    def forward(self, batch):
        try:
            float_input = batch["audio_data"].squeeze(1)  # remove channel dimension
            with autocast(
                float_input.device.type,
                enabled=False,
            ):
                mel_spec = self.feature_extractor(float_input)[:, None]
        except RuntimeError as e:
            print(batch["audio_data"].shape, float_input.shape)
            raise e
        if "logits" in batch:
            teacher_logits = batch["logits"]
            teacher_feats = None
        else:
            with torch.no_grad():
                teacher_logits, teacher_feats = self.teacher(mel_spec.detach())
        student_logits, student_feats = self.student(mel_spec)
        return student_logits, student_feats, teacher_logits, teacher_feats, mel_spec

    def predict(self, batch):
        float_input = batch["audio_data"].squeeze(1)
        mel_spec = self.feature_extractor(float_input)[:, None]
        with torch.no_grad():
            student_logits, _ = self.student(mel_spec)
            pred = torch.softmax(student_logits, dim=-1)
        return pred

    def compute_loss(self, inputs, outputs):
        student_logits, student_feats, teacher_logits, teacher_feats, mel_spec = outputs
        targets = inputs["weak_targets"]
        
        label_targets = targets
        label_logits = student_logits
        
        teacher_act = torch.sigmoid(teacher_logits)
        kd_loss_value = self.kd_loss(student_logits, teacher_act)
        label_loss_value = self.label_loss(label_logits, label_targets)

        loss = (
            1 - self.label_loss_prop
        ) * kd_loss_value + self.label_loss_prop * label_loss_value
        # scalars
        summary = {
            "scalars": {
                "kd_loss": kd_loss_value,
                "label_loss": label_loss_value,
            },
        }
        return loss, targets, summary
        

class LogitDistillationModel(AbstractDistillationModel):
    """Response-based distillation model.

    Loss is computed between soft score of teacher and soft score of student.
    Teacher model is not needed, we only use teacher ensemble logits provided by the dataset
    """

    def __init__(
        self,
        student: nn.Module,
        feature_extractor=None,
        label_loss_prop=0.1,
        kd_loss=torch.nn.BCEWithLogitsLoss(),
        label_loss=torch.nn.BCEWithLogitsLoss(),
        projection_layer=None,
    ):
        """
        Args:
            teacher: teacher model
        """
        super().__init__()
        self.student = student
        # ignore teacherparam
        self.feature_extractor = feature_extractor
        self.label_loss_prop = label_loss_prop
        self.kd_loss = kd_loss
        self.label_loss = label_loss

        self.labelwise_metrics = ()
        self.label_mapping = None
        self.projection_layer = projection_layer


        print(
            f"Student Parameters: {sum([p.numel() for p in self.student.parameters()])}"
        )

    def forward(self, batch):
        try:
            float_input = batch["audio_data"].squeeze(1)
            with autocast(
                float_input.device.type,
                enabled=False,
            ):
                mel_spec = self.feature_extractor(float_input)[:, None]
        except RuntimeError as e:
            print(batch["audio_data"].shape, float_input.shape)
            raise e
        student_logits, student_features = self.student(mel_spec)

        return student_logits, student_features, mel_spec


    def predict(self, batch):
        float_input = batch["audio_data"].squeeze(1)
        mel_spec = self.feature_extractor(float_input)
        with torch.no_grad():
            student_logits, _ = self.student(mel_spec)
            pred = torch.softmax(student_logits, dim=-1)
        return pred

    def compute_loss(self, inputs, outputs):
        student_logits, student_features, _ = outputs

        assert "logits" in inputs, inputs.keys()
        teacher_logits = inputs["logits"]
        if teacher_logits.dtype != student_logits.dtype:
            teacher_logits = teacher_logits.to(student_logits.dtype)
        if isinstance(self.label_loss, nn.BCEWithLogitsLoss):
            targets = inputs["weak_targets"]
        elif isinstance(self.label_loss, nn.CrossEntropyLoss):
            targets = inputs["target"]
        else:
            raise ValueError(f"Unknown label loss {self.label_loss}")
        
        label_loss_value = self.label_loss(student_logits, targets)

        teacher_act = torch.sigmoid(teacher_logits)

        if self.projection_layer is None:
            kd_loss_value = self.kd_loss(student_logits, teacher_act)
        else:
            kd_loss_value = self.kd_loss(self.projection_layer(student_features).reshape(teacher_act.shape), teacher_act)
        loss = (
            1 - self.label_loss_prop
        ) * kd_loss_value + self.label_loss_prop * label_loss_value
        # scalars
        summary = {
            "scalars": {
                "kd_loss": kd_loss_value,
                "label_loss": label_loss_value,
            },
        }
        return loss, targets, summary
