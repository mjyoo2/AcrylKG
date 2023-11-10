from datasets import load_dataset, load_metric
from transformers import AutoTokenizer

import transformers
import torch
import numpy as np
import tqdm

dataset = load_dataset('qa_srl', split='test')
metric = load_metric('accuracy', 'f1')

print(dataset)

model = "meta-llama/Llama-2-7b-chat-hf"
# model = "TheBloke/Llama-2-70b-chat-GPTQ"
tokenizer = AutoTokenizer.from_pretrained(model)

# quantization_config = GPTQConfig(

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto", # if you have GPU
    # do_sample=True,
    # top_k=10,
    # top_p=0.9,
    temperature=1.0,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=500, # can increase the length of sequence
)

# context = "Document : John works at Google. Google is a tech company. John, who is an engineer, designed a bridge that spans the wide river, and he worked with a team.\nKnowledge Graph : (John, works-at, Google), (Google, is, tech-company), (John, is, engineer), (John, designed, bridge), (bridge, spans, river), (John, worked-with, team)\n"

def precision_and_recall(reference, prediction):
    """Calculate word-level precision and recall between two sentences, considering duplicates."""

    reference_words = reference.lower().split()
    prediction_words = prediction.lower().split()

    TP = sum(word in reference_words for word in prediction_words)
    FP = sum(word not in reference_words for word in prediction_words)
    FN = sum(word not in prediction_words for word in reference_words)

    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0

    return precision, recall

instruction = "Using a contiguous segment from the provided document, please generate a concise response. Only include essential information.\n"

example_1 = """
Document: There are four boat clubs that row on the River Dee : Aberdeen Boat Club , Aberdeen Schools Rowing Association , Aberdeen University Boat Club and Robert Gordon University Boat Club.
Question: where does something row ?
Answer: on the River Dee
"""

example_2 = """
Document: The proposal intends to use wind-driven pumps to inject oxygen into waters at , or around , 130m below sea level .
Question: what will something be used ?
Answer: to inject oxygen into waters
"""

example_3 = """
Document: Valdano added that `` Maradona offered to the Argentines a way out of their collective frustration , and that 's why people love him .
Question: what did someone add ?
Answer: that `` Maradona offered to the Argentines a way out of their collective frustration , and that 's why people love him
"""

example_4 = """
Document: In the definitions offered by Beauchamp & Davidson and , later , by Wreen , consent on the part of the patient was not considered to be one of their criteria , although it may have been required to justify euthanasia .
Question: who was something offered by ?
Answer: Beauchamp & Davidson and , later , by Wreen
"""

example_5 = """
Document: Quill pens were used to write the vast majority of medieval manuscripts , the Magna Carta and the Declaration of Independence .
Question: what was used ?
Answer: Quill pens
"""

prompt = lambda x, y: example_1 + example_2 + example_3 + example_4 + example_5 + '\n' + "Document: {}\nQuestion: {}\nAnswer:".format(x, y)
precision_list, recall_list = [], []
tbar = tqdm.tqdm(range(2201))
for i in tbar:
    query = dataset[i]
    document = query['sentence']
    _question = query['question']
    question = []
    for word in _question:
        if word != '_':
            question.append(word)

    question = ' '.join(question)

    sequences = pipeline(prompt(document, question))
    llm_answer = (sequences[0]['generated_text'].split('\n'))[23][8:]
    p, r = 0, 0
    for answer in query['answers']:
        _p, _r = precision_and_recall(answer, llm_answer)
        if p + r < _p + _r:
            p = _p
            r = _r
    precision_list.append(p)
    recall_list.append(r)
    tbar.set_description("Precision: {} Recall: {} F1 score: {}".format(np.mean(precision_list), np.mean(recall_list), 2 * (np.mean(precision_list) * np.mean(recall_list)) / (np.mean(precision_list) + np.mean(recall_list) + 1e-10)))
