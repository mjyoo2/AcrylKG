from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig
import transformers
import torch
import os
import argparse
import time
from description import BILL_GATES, BILL_GATES_RDF

# os.environ["HF_HOME"] = "/workspace/models"

parser = argparse.ArgumentParser(description="Main training script for the model")
parser.add_argument(
    "--num-parameters",
    type=str,
    default="7b"
)
args = parser.parse_args()

# Hugging face repo name
model = "meta-llama/Llama-2-{}-chat-hf".format(args.num_parameters) #chat-hf (hugging face wrapper version)
# model = "TheBloke/Llama-2-{}-chat-GPTQ".format(args.num_parameters) #chat-hf (hugging face wrapper version)

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto" # if you have GPU
)

text = "Type apple once."
sequences = pipeline(
    text,  # + '\nRequest: Explain people related to the person in this description. \n Answer:',
    do_sample=True,
    top_k=10,
    top_p=0.9,
    temperature=0.2,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=6,  # can increase the length of sequence
)

for test_tokens in [1, 3, 10, 20, 50, 100, 200, 500]:
    text = "Type apple once." * test_tokens

    start_time = time.time()
    sequences = pipeline(
        text, #+ '\nRequest: Explain people related to the person in this description. \n Answer:',
        do_sample=True,
        top_k=10,
        top_p=0.9,
        temperature=0.2,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=test_tokens * 4 + 2, # can increase the length of sequence
    )
    print(time.time() - start_time)

# next_query = ''
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
    # next_query = next_query + seq['generated_text']