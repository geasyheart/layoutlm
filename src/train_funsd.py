# -*- coding: utf8 -*-
#
from typing import Dict

from torch import nn
from utrainer import UTrainer

from src.g import get_ptm_model
from src.metric import LayoutLMMetric
from src.transform import FunsdDataSet, get_labels


class FunsdLayoutLMTrainer(UTrainer):

    def train_steps(self, batch_idx, batch_data) -> Dict:
        metric = LayoutLMMetric()
        metric.step(self.evaluate_steps(batch_idx=batch_idx, batch_data=batch_data))
        metric.score()
        input_ids, token_type_ids, attention_mask, bbox, labels = batch_data
        outputs = self.model(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )
        if isinstance(self.model, nn.DataParallel):
            loss = outputs.loss.mean()
        else:
            loss = outputs.loss
        return {"loss": loss}

    def evaluate_steps(self, batch_idx, batch_data):
        input_ids, token_type_ids, attention_mask, bbox, labels = batch_data
        outputs = self.model(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )
        y_preds, y_trues = outputs.logits.argmax(-1).tolist(), labels.tolist()

        new_y_preds, new_y_trues = [], []
        for y_pred, y_true in zip(y_preds, y_trues):
            new_y_pred, new_y_true = [], []
            for yp, yt in zip(y_pred, y_true):
                if yt != -100:
                    new_y_pred.append(yp)
                    new_y_true.append(yt)
            new_y_preds.append(new_y_pred)
            new_y_trues.append(new_y_true)
        return new_y_preds, new_y_trues


if __name__ == '__main__':
    trainer = FunsdLayoutLMTrainer()

    trainer.model = nn.DataParallel(get_ptm_model(num_labels=len(get_labels())))

    batch_size = 16
    train_dl = FunsdDataSet(mode='train', device=trainer.device).to_dl(batch_size=batch_size, shuffle=True)
    dev_dl = FunsdDataSet(mode='test', device=trainer.device).to_dl(batch_size=batch_size, shuffle=False)

    trainer.fit(
        train_dl=train_dl,
        dev_dl=dev_dl,
        metric_cls=LayoutLMMetric
    )
