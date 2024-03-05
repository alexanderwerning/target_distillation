import numpy as np
import padertorch as pt
import torch
import torch.nn as nn
from pb_sed.evaluation import instance_based
from sklearn import metrics
from sklearn.decomposition import PCA
from torch import autocast
from torchvision.utils import make_grid
import warnings

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

        top1acc = (y.argmax(-1) == targets.argmax(-1)).mean()
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
            with autocast(mel_spec.device.type):  # does this break on cpu?
                logits, _ = self.net(mel_spec)
                pred = torch.softmax(logits, dim=-1)
        return pred

    def compute_loss(self, inputs, outputs):
        logits, feats, mel_spec = outputs
        targets = inputs["weak_targets"]
        if isinstance(self.loss_fn, torch.nn.CrossEntropyLoss) and targets.dim() == 2:
            targets = torch.argmax(targets, dim=-1).long()
        loss = self.loss_fn(logits, targets)

        # scalars
        summary = {
            "scalars": {
                "loss": loss,
            },
        }
        return loss, summary

    def review(self, inputs, outputs):
        loss, loss_summary = self.compute_loss(inputs, outputs)
        logits = outputs[0]
        mel_spec = outputs[-1]

        targets = inputs["weak_targets"]
        labeled_examples_idx = (
            ((targets < 0.00001) + (targets > 0.99999)).detach().cpu().numpy() == 1
        ).all(-1)
        y_weak = logits.detach().cpu().numpy()[labeled_examples_idx]
        weak_targets = targets.detach().cpu().numpy()[labeled_examples_idx]
        summary = {
            "loss": loss,
            "scalars": {},
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


class DistillationModel(pt.Model):
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
            with autocast(mel_spec.device.type):
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
        return loss, summary

    def review(self, inputs, outputs):
        loss, loss_summary = self.compute_loss(inputs, outputs)
        student_logits = outputs[0]
        mel_spec = outputs[-1]

        targets = inputs["weak_targets"]
        labeled_examples_idx = (
            ((targets < 0.00001) + (targets > 0.99999)).detach().cpu().numpy() == 1
        ).all(-1)

        y_weak = student_logits.detach().cpu().numpy()[labeled_examples_idx]
        weak_targets = targets.detach().cpu().numpy()[labeled_examples_idx]
        summary = {
            "loss": loss,
            "scalars": {},
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
        

class EmbeddingClassifier(pt.Model):
    def __init__(
        self, input_dim=768, pca_n_components=768, n_classes=50, hidden_dim=1000
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(pca_n_components, hidden_dim),
            nn.Hardswish(),
            nn.Linear(hidden_dim, n_classes),
        )
        if input_dim == pca_n_components:
            self.pca_layer = nn.Identity()
            self.pca = None
        else:
            self.pca_layer = nn.Linear(input_dim, pca_n_components)
            self.pca_layer.weight.requires_grad = False
            self.pca_layer.bias.requires_grad = False
            self.pca = PCA(n_components=pca_n_components)

    def fit(self, train_dataset, device=None):
        """Fit the representation transform on the training dataset"""
        if isinstance(self.pca, PCA):
            embeddings = [batch["feature"] for batch in train_dataset]
            embeddings = np.concatenate(embeddings)
            self.pca.fit(embeddings)
            self.pca_layer.weight.data = torch.from_numpy(self.pca.components_)
            self.pca_layer.bias.data = torch.from_numpy(
                self.pca.components_ @ self.pca.mean_
            )

    def forward(self, inputs):
        reduced_feats = self.pca_layer(inputs["feature"])
        logits = self.classifier(reduced_feats)
        pred = torch.softmax(logits, dim=-1)
        return pred, logits

    def predict(self, inputs):
        with torch.no_grad():
            return self.forward(inputs)[0]

    def review(self, inputs, outputs):
        pred, _ = outputs
        labels = inputs["weak_targets"]
        loss = nn.CrossEntropyLoss()(pred, labels)
        summary = {
            "loss": loss,
            "buffers": {
                "y_weak": pred.detach().cpu().numpy(),
                "targets_weak": labels.detach().cpu().numpy(),
            },
        }
        return summary

    def modify_summary(self, summary):
        """called by the trainer before dumping a summary"""
        if f"targets_weak" in summary["buffers"]:
            # Computes fscores from scores and targets
            self.add_metrics_to_summary(summary, "weak")
        summary = super().modify_summary(summary)
        return summary

    def add_metrics_to_summary(self, summary, suffix):
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

        top1acc = (y.argmax(-1) == targets.argmax(-1)).mean()
        summary["scalars"][f"top1acc_{suffix}"] = top1acc


class BlendedDistillationModel(pt.Model):
    """Response-based distillation model.

    Loss is computed between soft score of teacher and soft score of student.
    Teacher model is not needed, we only use teacher ensemble logits provided by the dataset
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module = None,
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
        self.student = student
        # ignore teacherparam
        self.feature_extractor = feature_extractor
        self.label_loss_prop = label_loss_prop
        self.kd_loss = kd_loss
        self.label_loss = label_loss

        self.labelwise_metrics = ()
        self.label_mapping = None

        # self._embedding_value = None

        # def get_input(model, input, output):
        #     self._embedding_value = input[0]  # unpack tuple

        # self.student.model.classifier[-2].register_forward_hook(get_input)

        hidden_size = self.student.model.classifier[-1].in_features
        self.teacher_projection = nn.Linear(hidden_size, 527)

        print(
            f"Student Parameters: {sum([p.numel() for p in self.student.parameters()])}, hidden_size: {hidden_size}"
        )

    def forward(self, batch):
        print(batch['audio_data'].shape)
        try:
            float_input = batch["audio_data"].squeeze(1)
            mel_spec = self.feature_extractor(float_input)
        except RuntimeError as e:
            print(batch["audio_data"].shape, float_input.shape)
            raise e
        with autocast(mel_spec.device.type):
            # student_logits, student_feats = self.student(mel_spec)
            student_logits, student_embedding = self.student.forward_return_emb(
                mel_spec
            )
            # pred_teacher_logits = self.teacher_projection(self._embedding_value)
            pred_teacher_logits = self.teacher_projection(student_embedding)
            student_feats = None

        return student_logits, student_feats, pred_teacher_logits, mel_spec

    def predict(self, batch):
        float_input = batch["audio_data"].squeeze(1)
        mel_spec = self.feature_extractor(float_input)
        with torch.no_grad():
            with autocast(mel_spec.device.type):
                student_logits, _ = self.student(mel_spec)
                pred = torch.softmax(student_logits, dim=-1)
        return pred

    def compute_loss(self, inputs, outputs):
        student_logits, student_feats, pred_teacher_logits, mel_spec = outputs

        assert "logits" in inputs, inputs.keys()
        teacher_logits = inputs["logits"]
        if isinstance(self.label_loss, nn.BCEWithLogitsLoss):
            targets = inputs["weak_targets"]
        elif isinstance(self.label_loss, nn.CrossEntropyLoss):
            targets = inputs["target_idx"]
        else:
            raise ValueError(f"Unknown label loss {self.label_loss}")
        label_targets = targets
        label_logits = student_logits

        with autocast(mel_spec.device.type):
            if len(label_targets) > 0:
                label_loss_value = self.label_loss(label_logits, label_targets)
            else:
                warnings.warn("Batch contains no labeled examples")
                label_loss_value = torch.zeros(
                    [], device=mel_spec.device, dtype=mel_spec.dtype
                )
            teacher_act = torch.sigmoid(teacher_logits)
            assert pred_teacher_logits.shape == teacher_act.shape, (
                pred_teacher_logits.shape,
                teacher_act.shape,
            )
            kd_loss_value = self.kd_loss(pred_teacher_logits, teacher_act)

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
        return loss, summary

    def review(self, inputs, outputs):
        loss, loss_summary = self.compute_loss(inputs, outputs)
        student_logits = outputs[0]
        mel_spec = outputs[-1]

        targets = inputs["weak_targets"]
        hard_target = (
            ((targets < 0.00001) + (targets > 0.99999)).detach().cpu().numpy() == 1
        ).all(-1)
        labeled_examples_idx = hard_target
        y_weak = student_logits.detach()[labeled_examples_idx].cpu().numpy()
        weak_targets = targets.detach()[labeled_examples_idx].cpu().numpy()
        summary = {
            "loss": loss,
            "scalars": {},
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