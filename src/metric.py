# -*- coding: utf8 -*-
#
from typing import Dict

from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from utrainer.metric import Metric

from src.transform import get_labels


class LayoutLMMetric(Metric):
    def __init__(self):
        self.label_map = {v: i for i, v in enumerate(get_labels())}
        self.id_label_map = {v: k for k, v in self.label_map.items()}

        self.y_preds = []
        self.y_trues = []

    def step(self, inputs):
        y_preds, y_trues = inputs
        for y_pred, y_true in zip(y_preds, y_trues):
            y_pred = [self.id_label_map[i] for i in y_pred]
            y_true = [self.id_label_map[i] for i in y_true]
            self.y_preds.append(y_pred)
            self.y_trues.append(y_true)

    def score(self) -> float:
        return f1_score(y_true=self.y_trues, y_pred=self.y_preds)

    def report(self) -> Dict:
        print(classification_report(y_true=self.y_trues, y_pred=self.y_preds))
        return {
            "precision": precision_score(y_true=self.y_trues, y_pred=self.y_preds),
            "recall": recall_score(y_true=self.y_trues, y_pred=self.y_preds),
            "f1": f1_score(y_true=self.y_trues, y_pred=self.y_preds)
        }
