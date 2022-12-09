# -*- coding: utf8 -*-
# https://huggingface.co/docs/transformers/v4.25.1/en/model_doc/layoutlm#transformers.LayoutLMForTokenClassification
from transformers import AutoTokenizer, LayoutLMConfig, LayoutLMForTokenClassification

pretrained_name = "microsoft/layoutlm-base-uncased"
# pretrained_name = 'microsoft/layoutlmv3-base'

tokenizer = AutoTokenizer.from_pretrained(pretrained_name)


def get_ptm_model(num_labels: int):
    config = LayoutLMConfig.from_pretrained(pretrained_name)
    config.num_labels = num_labels
    ptm_model = LayoutLMForTokenClassification.from_pretrained(pretrained_name, config=config)
    return ptm_model
