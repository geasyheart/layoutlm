# -*- coding: utf8 -*-


# https://huggingface.co/docs/transformers/main/en/model_doc/layoutxlm#overview
# https://huggingface.co/docs/transformers/main/en/model_doc/layoutlmv2#overview

from transformers import LayoutXLMTokenizer, LayoutLMv2Config, LayoutLMv2ForTokenClassification

pretrained_name = "microsoft/layoutxlm-base"

tokenizer = LayoutXLMTokenizer.from_pretrained(pretrained_name)


def get_ptm_model(num_labels: int):
    config = LayoutLMv2Config.from_pretrained(pretrained_name)
    config.num_labels = num_labels
    ptm_model = LayoutLMv2ForTokenClassification.from_pretrained(pretrained_name, config=config)
    return ptm_model
