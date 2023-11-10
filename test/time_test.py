import torch
import transformers
import gradio as gr
import chromadb
import numpy as np
import time
import matplotlib as plt
import random
import string
from transformers import set_seed
# set_seed(777)

from langchain.chains import ConversationChain, SequentialChain, TransformChain, LLMChain
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import Document
from typing import List
from langchain.chains import LLMChain
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

# from prompt.kg_template import (
#     get_base_prompt,
#     get_prompt_1,
#     get_prompt_2,
#     get_fewshot_prompt
# )
# from viz.graph_viz import graph_visualize

small_llm = "meta-llama/Llama-2-7b-chat-hf"
large_llm = "TheBloke/Llama-2-70b-chat-GPTQ"
max_length = 2048


# Output parser will split the LLM result into a list of queries
class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")


class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)


def construct_kg_chain():
    kg_prompt = PromptTemplate(
        input_variables=["input_text"],
        template="{input_text}",
    )

    # Load knowledge graph models
    tokenizer = AutoTokenizer.from_pretrained(small_llm)
    hf_llm_pipeline = transformers.pipeline(
        "text-generation",
        model=large_llm,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
        max_length=max_length,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )

    llm_pipeline = HuggingFacePipeline(pipeline=hf_llm_pipeline, model_kwargs={'temperature': 0.7})
    kg_chain = LLMChain(llm=llm_pipeline, prompt=kg_prompt, output_key="knowledge_graph", verbose=True)
    return kg_chain

def random_string(length):
    """Return a random string of specified length."""
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))

def generate_random_string_list(N, min_length=5, max_length=10):
    """Return a list of N random strings."""
    return [random_string(random.randint(min_length, max_length)) for _ in range(N)]

def construct_retrival_fn(retriever):

    start_time = time.time()
    for i in range(1000):
        retrieved_docs = retriever.invoke("What did the president say about Ketanji Brown Jackson?")
    spend_time = time.time() - start_time
    return spend_time

def visualize_execution_time():
    x_values = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 40000]
    y_values = [200, 500, 1000, 2000]#[1, 2, 5, 10, 20, 50, 100]
    execution_times = np.zeros((len(x_values), len(y_values)))

    for i, x in enumerate(x_values):
        kg_output = generate_random_string_list(x, max_length=50, min_length=10)
        knowledge_graph_docs = [Document(page_content=doc) for doc in kg_output]
        DPR_model = HuggingFaceEmbeddings(model_name='sentence-transformers/facebook-dpr-question_encoder-single-nq-base')
        vectordb = Chroma.from_documents(knowledge_graph_docs, DPR_model)
        for j, y in enumerate(y_values):
            retriever = vectordb.as_retriever(search_kwargs={'k': y})
            if x > y:
                execution_times[i][j] = construct_retrival_fn(retriever)
                print("Num_KG: {} Num_search: {} Exec time: {}".format(x, y, execution_times[i][j]))
    plt.imshow(execution_times, cmap='hot', interpolation='nearest', extent=[y_values[0], y_values[-1], x_values[0], x_values[-1]])
    plt.colorbar(label='Execution Time (s)')
    plt.xlabel('Y Values')
    plt.ylabel('X Values')
    plt.title('Execution Time Visualization')
    plt.gca().invert_yaxis()  # y축을 내림차순으로 변경
    plt.savefig('exec_time.png')


# 실행 예시

if __name__ == '__main__':
    llm_chain = construct_kg_chain()
    print(llm_chain.run({'input_text': "Used: Repeat apple\n" + "apple" * 5}))


    # visualize_execution_time()
    # construct_retrival_fn(['a', 'b', 'c'])