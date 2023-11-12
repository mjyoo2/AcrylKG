import torch
import transformers
import gradio as gr
import chromadb
import numpy as np
from transformers import set_seed
import wikipediaapi
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


from prompt.kg_template import (
    get_base_prompt,
    get_prompt_1,
    get_prompt_2,
    get_fewshot_prompt,
    get_fewsot_prompt_QA
)
from viz.graph_viz import graph_visualize


# LLM model
# llm = "meta-llama/Llama-2-7b-chat-hf"
# llm = "gpt2-medium"
llm = "TheBloke/Llama-2-70b-chat-GPTQ"
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
        input_variables=["context", "input_text"],
        template="Generate knowledge graph for the given document.\n{context}\nDocument : {input_text}\nKnowledge Graph : ",
    )

    # Load knowledge graph models
    tokenizer=AutoTokenizer.from_pretrained(llm)
    hf_llm_pipeline=transformers.pipeline(
        "text-generation",
        model=llm,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
        max_length=max_length ,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )

    llm_pipeline = HuggingFacePipeline(pipeline=hf_llm_pipeline, model_kwargs={'temperature':0.7})
    kg_chain = LLMChain(llm=llm_pipeline, prompt=kg_prompt, output_key="knowledge_graph",  verbose=True)
    return kg_chain

def construct_retrival_qa_chain(kg_output, llm_pipeline, prompt):
    kg_output = kg_output.split("\n")[0]
    kg_html, triples = graph_visualize(kg_output, ".")
    
    # output_parser = LineListOutputParser()
    # kg_output = output_parser.parse(kg_output)

    knowledge_graph_docs = [Document(page_content='(' + ', '.join(doc) + ')') for doc in triples]
    DPR_model = HuggingFaceEmbeddings(model_name='sentence-transformers/facebook-dpr-question_encoder-single-nq-base')
    vector_db = Chroma.from_documents(knowledge_graph_docs, DPR_model)

    chain_type_kwargs = {"prompt": prompt, "verbose": True}
    retriever = RetrievalQA.from_chain_type(retriever=vector_db.as_retriever(search_kwargs={'k': 5}), llm=llm_pipeline, chain_type_kwargs=chain_type_kwargs)
    return retriever, kg_html

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
    llm_pipeline = HuggingFacePipeline(pipeline=hf_llm_pipeline, model_kwargs={'temperature': 0.7})

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

def remove_non_ascii(s):
    return "".join(i for i in s if ord(i) < 128)

if __name__ == '__main__':
    wiki_en = wikipediaapi.Wikipedia(
        "mjyoo2 (mjyoo2@skku.edu)",
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI
    )

    wiki_page = wiki_en.page("Kim Ku")
    docs = remove_non_ascii(wiki_page.text[:5000]).replace('\n', ' ')
    del wiki_en

    kg_chain = construct_kg_chain()
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
        eos_token_id=tokenizer.eos_token_id,
        # load_in_8bit=True,
    )

    llm_pipeline = HuggingFacePipeline(pipeline=hf_llm_pipeline, model_kwargs={'temperature': 0.7})
    input_query, context = get_fewshot_prompt()
    kg_output = kg_chain.run({'context': context, 'input_text': docs})
    print(kg_output)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="System: Answer only based on context. Do not use common knowledge. \nContext: {context}.\nQuestion : {question}\nAnswer :",
    )

    # cot_prompt = PromptTemplate(
    #     input_variables=["context", "question"],
    #     template="System: Answer only based on context. Do not use common knowledge.\n" + get_fewsot_prompt_QA() + "Context: {context}.\nQuestion: {question}\nAnswer:",
    # )

    retrival_qa_chain, kg_html = construct_retrival_qa_chain(kg_output, llm_pipeline, cot_prompt)
    for i in range(5):
        question = input()
        answer = retrival_qa_chain.run(question)
        print(answer)

    # def kg_fn(document, question):
    #     input_query, context = get_fewshot_prompt()
    #     kg_output = kg_chain.run({'context': context, 'input_text': document})
    #     retrival_qa_chain, kg_html = construct_retrival_qa_chain(kg_output, llm_pipeline)
    #     answer = retrival_qa_chain.run(question)
    #     return kg_output, kg_html, answer
    #
    # demo = gr.Interface(
    #     fn=kg_fn,
    #     inputs=["text", "text"],
    #     outputs=["text", "html", "text"],
    # )
    # demo.queue().launch(debug=True, share=True, inline=False)