import tqdm
import torch
import transformers
import gradio as gr
import chromadb
import numpy as np
import pickle as pkl
import re
from transformers import set_seed
import wikipediaapi
import os
import argparse
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
from langchain.document_loaders import TextLoader

import kimku_wiki
from prompt.kg_template import (
    get_fewshot_prompt_detailed_ko,
    get_fewshot_prompt,
    get_fewsot_prompt_QA,
    get_fewshot_prompt_detailed
)
from viz.graph_viz import graph_visualize

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

def construct_kg_chain(llm_pipeline):
    kg_prompt = PromptTemplate(
        input_variables=["context", "input_text"],
        template="Generate knowledge graph for the given document.\n{context}\nDocument : {input_text}\nKnowledge Graph :",
    )
    kg_chain = LLMChain(llm=llm_pipeline, prompt=kg_prompt, output_key="knowledge_graph",  verbose=True)
    return kg_chain

def construct_retrival_qa_chain(knowledge_graph_docs, llm_pipeline, prompt):
    DPR_model = HuggingFaceEmbeddings(model_name=sentence_embedding_model)
    vector_db = Chroma.from_documents(knowledge_graph_docs, DPR_model)

    chain_type_kwargs = {"prompt": prompt, "verbose": True}
    retriever = RetrievalQA.from_chain_type(retriever=vector_db.as_retriever(search_kwargs={'k': 5}), llm=llm_pipeline, chain_type_kwargs=chain_type_kwargs, return_source_documents=True)
    return retriever

def remove_non_ascii(s):
    return "".join(i for i in s if ord(i) < 128)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--llm', type=str, default='LLAMA_70B')
    args = parser.parse_args()

    if args.llm == 'LLAMA_70B':
        llm = "TheBloke/Llama-2-70b-chat-GPTQ"
        max_length = 1024
        sentence_embedding_model = 'sentence-transformers/facebook-dpr-question_encoder-single-nq-base'
        get_prompt = get_fewshot_prompt_detailed
    elif args.llm == 'LLAMA_7B':
        llm = "meta-llama/Llama-2-7b-chat-hf"
        max_length = 1024
        sentence_embedding_model = 'sentence-transformers/facebook-dpr-question_encoder-single-nq-base'
        get_prompt = get_fewshot_prompt_detailed
    elif args.llm == "POLYGLOT":
        llm = "EleutherAI/polyglot-ko-12.8b"
        max_length = 2048
        sentence_embedding_model = "jhgan/ko-sroberta-multitask"
        get_prompt = get_fewshot_prompt_detailed_ko
    elif args.llm == 'LLAMA_70B_KO':
        llm = "quantumaikr/llama-2-70b-fb16-korean"
        max_length = 1024
        sentence_embedding_model = "jhgan/ko-sroberta-multitask"
        get_prompt = get_fewshot_prompt_detailed_ko
    else:
        raise NotImplementedError()

    docs = kimku_wiki.ko.replace('\n', ' ')
    tokenizer = AutoTokenizer.from_pretrained(llm)
    hf_llm_pipeline = transformers.pipeline(
        "text-generation",
        model=llm,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
        max_length=max_length,
        do_sample=False,
        # top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        load_in_4bit=True,
    )

    llm_pipeline = HuggingFacePipeline(pipeline=hf_llm_pipeline, model_kwargs={'temperature': 1.})
    kg_chain = construct_kg_chain(llm_pipeline)
    input_query, context = get_prompt()

    if not os.path.exists('./cache/knowledge_graph_{}.pkl'.format(LLM)):
        docs_sentences = re.split(r'\. ?', docs)[:-1]

        print(docs_sentences)
        knowledge_graph_docs = []
        knowledge_graph_str_docs = []
        for sen_idx in range(len(docs_sentences)//3 + 1):
            sentence = docs_sentences[sen_idx * 3: sen_idx * 3 + 3]
            kg_output = kg_chain.run({'context': context, 'input_text': '. '.join(sentence) +'.'})
            print(kg_output)
            for i in range(10):
                kg_output = kg_output.split("\n")[i]
                kg_html, triples = graph_visualize(kg_output, ".")
                if len(triples) > 0:
                    break
            knowledge_graph_docs.extend(triples)
        with open('./cache/knowledge_graph_{}.pkl'.format(LLM), 'wb') as f:
            pkl.dump(knowledge_graph_docs, f)
    else:
        with open('./cache/knowledge_graph_{}.pkl'.format(LLM), 'rb') as f:
            knowledge_graph_docs = pkl.load(f)

    kg_html, _ =  graph_visualize(knowledge_graph_docs, ".")
    print(knowledge_graph_docs)
    knowledge_graph_docs = [Document(page_content='(' + ', '.join(doc) + ')') for doc in knowledge_graph_docs]
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="System: You are a virtual human. Answer only based on context. Do not use common knowledge. \nContext: {context}.\nQuestion : {question}\nAnswer :",
    )

    retrival_qa_chain = construct_retrival_qa_chain(knowledge_graph_docs, llm_pipeline, prompt)
    def kg_fn(question):
        answer = retrival_qa_chain(question)
        return docs, answer['source_documents'], kg_html, answer['result']

    demo = gr.Interface(
        fn=kg_fn,
        inputs=["text"],
        outputs=["text", "text", "html", "text"],
    )
    demo.queue().launch(debug=True, share=True, inline=False)