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
import openai
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

openai.api_key = os.getenv("OPENAI_API_KEY")
ENGINE = "gpt-3.5-turbo-instruct"  # "text-ada-001"

import kimku_wiki
from prompt.kg_template import (
    get_fewshot_prompt_detailed_ko,
    get_fewshot_prompt,
    get_fewsot_prompt_QA,
    get_fewshot_prompt_detailed
)
from viz.graph_viz import graph_visualize


LLM = "LLAMA_70B_KO"

if LLM == 'LLAMA_70B':
    llm = "TheBloke/Llama-2-70b-chat-GPTQ"
    max_length = 1600
    SENTENCE_EMBEDDING_MODEL = 'sentence-transformers/facebook-dpr-question_encoder-single-nq-base'
    get_prompt = get_fewshot_prompt_detailed
elif LLM == 'LLAMA_7B':
    llm = "meta-llama/Llama-2-7b-chat-hf"
    max_length = 1600
    SENTENCE_EMBEDDING_MODEL = 'sentence-transformers/facebook-dpr-question_encoder-single-nq-base'
    get_prompt = get_fewshot_prompt_detailed
elif LLM == "POLYGLOT":
    llm = "EleutherAI/polyglot-ko-12.8b"
    max_length = 2048
    SENTENCE_EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"
    get_prompt = get_fewshot_prompt_detailed_ko
elif LLM == 'LLAMA_70B_KO':
    llm = "beomi/llama-2-ko-70b"
    max_length = 1024
    SENTENCE_EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"
    get_prompt = get_fewshot_prompt_detailed_ko
else:
    raise NotImplementedError()


# LLM model
# llm = "meta-llama/Llama-2-7b-chat-hf"
# llm = "gpt2-medium"

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
        template="{context}Document : {input_text}\nKnowledge Graph :",
    )

    kg_chain = LLMChain(llm=llm_pipeline, prompt=kg_prompt, output_key="knowledge_graph",  verbose=True)
    return kg_chain

def ko_construct_kg_chain(llm_pipeline):
    kg_prompt = PromptTemplate(
        input_variables=["context", "input_text"],
        template="{context}Document : {input_text}\nKnowledge Graph :",
    )

    kg_chain = LLMChain(llm=llm_pipeline, prompt=kg_prompt, output_key="knowledge_graph",  verbose=True)
    return kg_chain


def construct_retrival_qa_chain(knowledge_graph_docs, llm_pipeline, prompt, sentence_embedding_model):
    DPR_model = HuggingFaceEmbeddings(model_name=sentence_embedding_model)
    vector_db = Chroma.from_documents(knowledge_graph_docs, DPR_model)

    chain_type_kwargs = {"prompt": prompt, "verbose": True}
    retriever = RetrievalQA.from_chain_type(retriever=vector_db.as_retriever(search_kwargs={'k': 5}), llm=llm_pipeline, chain_type_kwargs=chain_type_kwargs, return_source_documents=True)
    return retriever

def construct_retrival_fn(kg_output):
    tokenizer = AutoTokenizer.from_pretrained(llm)
    hf_llm_pipeline = transformers.pipeline(
        "text-generation",
        model=llm,
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
    llm_pipeline = HuggingFacePipeline(pipeline=hf_llm_pipeline, model_kwargs={'temperature': 1.})

    knowledge_graph_docs = [Document(page_content=doc) for doc in kg_output.lines]
    DPR_model = HuggingFaceEmbeddings(model_name='sentence-transformers/facebook-dpr-question_encoder-single-nq-base')
    vectordb = Chroma.from_documents(knowledge_graph_docs, DPR_model)

    output_parser = LineListOutputParser()
    retrival_prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate retrival prompt for the given user question to retrieve relevant documents from a vector database. 
    By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions for the user question.
    Original question: {question}""",
    )

    retrival_chain = LLMChain(llm=llm_pipeline, prompt=retrival_prompt, verbose=True, output_parser=output_parser)
    retriever = MultiQueryRetriever(retriever=vectordb.as_retriever(search_kwargs={'k': 2}), llm_chain=retrival_chain, parser_key="lines")

    unique_docs = retriever.get_relevant_documents(
        query="Please explain about the company related to Bill Gates."
    )

    print(unique_docs)

if __name__ == '__main__':
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
        temperature=0,
        # top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        load_in_8bit=True,
    )
    llm_pipeline = HuggingFacePipeline(pipeline=hf_llm_pipeline, model_kwargs={'temperature': 0.0})

    if not os.path.exists('./cache/knowledge_graph_multilingual_{}.pkl'.format(LLM)):
        kg_chain = construct_kg_chain(llm_pipeline)
        input_query, context = get_prompt()
        docs_sentences = re.split(r'\. ?', docs)[:-1]
        knowledge_graph_docs = []
        for sen_idx in range((len(docs_sentences) - 1)//5 + 1):
            sentence = docs_sentences[sen_idx * 5: sen_idx * 5 + 5]
            kg_output = kg_chain.run({'context': context, 'input_text': "This is the part of Kim Ku's wiki. " + '. '.join(sentence) +'.'})
            kg_output_lines = kg_output.split("\n")
            for i in range(10):
                output = kg_output_lines[i]
                kg_html, triples = graph_visualize(output, ".")
                if len(triples) > 0:
                    break
            knowledge_graph_docs.extend(triples)
        with open('./cache/knowledge_graph_multilingual_{}.pkl'.format(LLM), 'wb') as f:
            pkl.dump(knowledge_graph_docs, f)
    else:
        with open('./cache/knowledge_graph_multilingual_{}.pkl'.format(LLM), 'rb') as f:
            knowledge_graph_docs = pkl.load(f)

    knowledge_graph_docs_list = [Document(page_content='(' + ', '.join(doc) + ')') for doc in knowledge_graph_docs]
    ko_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="시스템: 당신은 가상 인간입니다. 주어진 정보 만을 사용하여 질문에 대답하십시오. 이미 알고 있는 상식을 사용하면 안됩니다. \n정보: {context}.\n질문 : {question}\n대답 :",
    )

    retrival_qa_chain = construct_retrival_qa_chain(knowledge_graph_docs_list, llm_pipeline, ko_prompt, "jhgan/ko-sroberta-multitask")

    def kg_fn(question):
        answer = retrival_qa_chain(question)
        answer_result = answer['result']
        retreival_kg = []
        for doc in answer['source_documents']:
            retreival_kg.append(doc.page_content)
        kg_html, _ = graph_visualize(knowledge_graph_docs, retreival_kg)
        print(answer_result)
        return docs, kg_html, "\n".join(retreival_kg), question, answer_result


    demo = gr.Interface(
        fn=kg_fn,
        inputs=[
            gr.Textbox(label="질문"),
        ],
        outputs=[
            gr.Textbox(label="주어진 문서"),
            gr.HTML(label="문서 기반 전체 지식 그래프"),
            gr.Textbox(label="질문에 의해 추출된 지식 그래프"),
            gr.Textbox(label="실제 질문 (입력 값)"),
            gr.Textbox(label="답변")
        ],
        examples=[
            ["김구는 언제 태어났습니까?"],
            ["누가 김구를 죽였나?"],
            ["김구의 아들에 대해 알려줘."],
            ["김구의 맥북과 관련된 일화에 대해 알려줘."]
        ]
    )
    demo.queue().launch(debug=True, share=True, inline=False)