# import torch
# from transformers import T5Tokenizer, T5ForConditionalGeneration

# model_id = 'megagonlabs/t5-base-japanese-web'
# tokenizer = T5Tokenizer.from_pretrained(model_id)
# model = T5ForConditionalGeneration.from_pretrained(model_id)
# sentence = 'うおおおおおおああああああ絵ケアかかかおおおおお。'
# inputs = tokenizer(sentence, return_tensors='pt')
# with torch.no_grad():
#     outputs = model(input_ids=inputs['input_ids'], labels=inputs['input_ids'])
# print(torch.exp(outputs.loss))

from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import numpy as np
from janome.tokenizer import Tokenizer


def score(model, tokenizer, sentence):
    tensor_input = tokenizer.encode(sentence, return_tensors='pt')
    repeat_input = tensor_input.repeat(tensor_input.size(-1) - 2, 1)
    mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
    masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
    labels = repeat_input.masked_fill(masked_input != tokenizer.mask_token_id, -100)
    with torch.inference_mode():
        loss = model(masked_input, labels=labels).loss
    return np.exp(loss.item())


def wakati(sentence):
    return ' '.join([token.surface for token in janome_tokenizer.tokenize(sentence)])


model_name = 'nlp-waseda/roberta-large-japanese'
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
janome_tokenizer = Tokenizer()

sentence = '早稲田大学で自然言語処理を研究する。'
print(wakati(sentence))
print(score(sentence=wakati(sentence), model=model, tokenizer=tokenizer))
sentence = '早稲田大学で自然言語処理を爆発する。'
print(wakati(sentence))
print(score(sentence=wakati(sentence), model=model, tokenizer=tokenizer))
