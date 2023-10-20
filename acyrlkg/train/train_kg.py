from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
import transformers
import torch
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from acyrlkg.train.utils import print_trainable_parameters
from acyrlkg.train.dataset import TextDataset

# Define the model ID for the sharded FALCON model by vilsonrodrigues
model_id = "meta-llama/Llama-2-7b-chat-hf"

# Configure BitsAndBytesConfig for 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Initialize the tokenizer using the model ID and set the pad token to be the same as the end of sentence token
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Initialize the pre-trained model using AutoModelForCausalLM
pretrained_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True,
)


pretrained_model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(pretrained_model)

config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)


#########################
#       Datasets        #
#########################

DATA_DIR = '../../data/'
data = load_dataset("json", data_files={"train": DATA_DIR + "TekGen/quadruples-train.json", "validation":  DATA_DIR + "TekGen/quadruples-validation.json"}, field="data")
tokenizer.pad_token = tokenizer.eos_token

train_dataset = data['train']
val_dataset = data['validation']


def get_labels(triples):
    a = []
    for triple in triples:
       a.append(f"({triple[0]}, {triple[1]}, {triple[2]})")
    return ", ".join(a)

train_label_dataset = train_dataset.map(
    lambda x: {
        "labels": get_labels(x['triples'])
    }
)
print(train_label_dataset)

train_encodings = tokenizer(
    train_dataset["sentence"],
    truncation=True,
    padding=True,
    max_length=256,
    return_tensors="pt",
)

train_label_encoding = tokenizer(
    train_label_dataset["labels"],
    truncation=True,
    padding=True,
    max_length=256,
    return_tensors="pt",
)

# Convert the encodings to PyTorch datasets
train_dataset = TextDataset(train_encodings, train_label_encoding)

#########################

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_dataset,
    # eval_dataset=val_dataset,
    args=transformers.TrainingArguments(
        num_train_epochs=10,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        warmup_ratio=0.05,
        max_steps=40,
        learning_rate=2.5e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()