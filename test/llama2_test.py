from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig
import transformers
import torch
import os
import argparse
from description import BILL_GATES, BILL_GATES_RDF

os.environ["HF_HOME"] = "/workspace/models"

parser = argparse.ArgumentParser(description="Main training script for the model")
parser.add_argument(
    "--num-parameters",
    type=str,
    default="70b"
)
args = parser.parse_args()

# Hugging face repo name
# model = "meta-llama/Llama-2-{}-chat-hf".format(args.num_parameters) #chat-hf (hugging face wrapper version)
model = "TheBloke/Llama-2-{}-chat-GPTQ".format(args.num_parameters) #chat-hf (hugging face wrapper version)

tokenizer = AutoTokenizer.from_pretrained(model)

# quantization_config = GPTQConfig(
#      bits=4,
#      group_size=128,
#      dataset="c4",
#      desc_act=False,
#      tokenizer=tokenizer
# )
# quant_model = AutoModelForCausalLM.from_pretrained(model, quantization_config=quantization_config, device_map='auto')

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto" # if you have GPU
)

text = BILL_GATES_RDF

sequences = pipeline(
    text + '\nRequest: Explain people related to the person in this description. \n Answer:',
    do_sample=True,
    top_k=10,
    top_p=0.9,
    temperature=0.2,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=1500, # can increase the length of sequence
)

# next_query = ''
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
    # next_query = next_query + seq['generated_text']

# sequences = pipeline(
#     next_query + '\nRequest: What are relations between the entities for constructing knowledge graph about the person in this description? Please construct detailed relation using only triples format like [Bill gates, born, 1955], [Bill gates, CEO, Microsoft]". Do not use 2-length tuple.\n Answer:',
#     do_sample=False,
#     top_k=10,
#     top_p=0.9,
#     temperature=0.,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id,
#     max_length=2000, # can increase the length of sequence
# )
#
# for seq in sequences:
#     print(f"Result: {seq['generated_text']}")