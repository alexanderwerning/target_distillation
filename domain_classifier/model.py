import torch

class NNModel:
    def __init__(self, clf, threshold=0.5):
        self.clf = clf
        self.threshold = threshold

    def predict(self, inputs):
        with torch.no_grad():
            out = self.clf(torch.as_tensor(inputs, dtype=torch.float32).cuda())
        score = out.detach().cpu().numpy().flatten()
        return (score > self.threshold)[:, None]