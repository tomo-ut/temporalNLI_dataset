from transformers import MT5ForConditionalGeneration, T5Tokenizer
import re


def compute_perplexity(masked_sentence, answers):
    input_ids = tokenizer(masked_sentence, return_tensors='pt').input_ids.to(device)
    labels = tokenizer(answers, return_tensors='pt').input_ids.to(device)
    output = model(input_ids=input_ids, labels=labels)
    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
    loss = loss_fct(output.logits.view(-1, output.logits.size(-1)), labels.view(-1))
    return torch.exp(loss[:-1].mean()).item(), list(zip(tokenizer.convert_ids_to_tokens(
        labels[0].cpu().detach().numpy().tolist())[:-1], loss.cpu().detach().numpy()[:-1]))


def predict_topk(text, k=10, max_len=20, beam_size=200):
    n_blank = text.count('<extra_id_')
    encoding = tokenizer.encode_plus(text, return_tensors="pt")
    input_ids = encoding["input_ids"]
    attention_masks = encoding["attention_mask"]

    beam_outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_masks,
        max_length=max_len,
        num_beams=beam_size,
        num_return_sequences=beam_size,
        early_stopping=True,
        output_scores=True,
        return_dict_in_generate=True
    )

    result = [tokenizer.decode(seq) for seq in beam_outputs['sequences']]

    unique_results = []
    seq_ids = []
    for i, res in enumerate(result):
        ans = extract_answers(res, n_blank)
        if len(ans) == n_blank and ans not in unique_results:
            unique_results.append(ans)
            seq_ids.append(i)
            if len(unique_results) == k:
                break

    return unique_results, beam_outputs, seq_ids


def extract_answers(seq, n_blank):
    # seq = seq.lstrip('<pad>').strip()
    pattern = r'>(.*?)<'
    answers = re.findall(pattern, seq)
    answers = [a for a in answers if a.strip()]
    answers_d = {}
    i = 0
    for ans in answers:
        if ans.strip():
            answers_d[f'<extra_id_{i}>'] = ans.strip()
            i += 1
            if i == n_blank:
                break
    # if not answers and '<extra_id_0>' in seq:
    #    answers_d['<extra_id_0>'] = seq.split('<extra_id_0>')[-1].strip()
    return answers_d


tokenizer = T5Tokenizer.from_pretrained("google/mt5-large")
model = MT5ForConditionalGeneration.from_pretrained("google/mt5-large")

# input_ids = tokenizer(
#     "スミス は 2時間 以内 に <extra_id_0> と <extra_id_1> と <extra_id_2> を 発見 した.",
#     return_tensors="pt").input_ids

# sequence_ids = model.generate(input_ids)
# sequences = tokenizer.batch_decode(sequence_ids)
# predict_topk("スミス は 2時間 以内 に <extra_id_0> の <extra_id_1> を 発見 した。")
sentences = ["私 は 2時間 以内 に <extra_id_0> の <extra_id_1> を 発見 した。",
             "私 は 2時間 以内 に <extra_id_0> と <extra_id_1> を 発見 した。",
             "スミス は 2時間 以内 に <extra_id_0> の <extra_id_1> を 発見 した。",
             "スミス は 2時間 以内 に <extra_id_0> と <extra_id_1> を 発見 した。"]

unique_results, beam_outputs, seq_ids = [], [], []
for sentence in sentences:
    unique_result, beam_output, seq_id = predict_topk(sentence)
    unique_results.append(unique_result)
    beam_outputs.append(beam_output)
    seq_ids.append(seq_id)
    print(sentence)
    for r in unique_result:
        print(r)
    print()
