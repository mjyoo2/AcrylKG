import torch
import transformers
import gradio as gr
import chromadb
import numpy as np

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
        template="Documents: {input_text} Question: Construct knowledge graph for the given text. Your answer should be a list of comma separated tuples of three strings as many as possible. Provide these tuples separated by newlines. Answer:",
    )

    # Load knowledge graph models
    tokenizer=AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    hf_llm_pipeline=transformers.pipeline(
        "text-generation",
        model="meta-llama/Llama-2-7b-chat-hf",
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=2000 ,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )

    llm_pipeline = HuggingFacePipeline(pipeline=hf_llm_pipeline, model_kwargs={'temperature':0.7})
    kg_chain = LLMChain(llm=llm_pipeline, prompt=kg_prompt, output_key="knowledge_graph",  verbose=True)
    return kg_chain

def construct_retrival_qa_chain(kg_output, llm_pipeline):
    output_parser = LineListOutputParser()
    kg_output = output_parser.parse(kg_output)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Q: Answer with {context}. {question}\nA:",
    )

    knowledge_graph_docs = [Document(page_content=doc) for doc in kg_output.lines]
    DPR_model = HuggingFaceEmbeddings(model_name='sentence-transformers/facebook-dpr-question_encoder-single-nq-base')
    vector_db = Chroma.from_documents(knowledge_graph_docs, DPR_model)

    chain_type_kwargs = {"prompt": prompt, "verbose": True}
    retriever = RetrievalQA.from_chain_type(retriever=vector_db.as_retriever(search_kwargs={'k': 2}), llm=llm_pipeline, chain_type_kwargs=chain_type_kwargs)
    return retriever

def construct_retrival_fn(kg_output):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    hf_llm_pipeline = transformers.pipeline(
        "text-generation",
        model="meta-llama/Llama-2-7b-chat-hf",
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=2000,
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

if __name__ == '__main__':
    kg_chain = construct_kg_chain()
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    hf_llm_pipeline = transformers.pipeline(
        "text-generation",
        model="meta-llama/Llama-2-7b-chat-hf",
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=2000,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )

    llm_pipeline = HuggingFacePipeline(pipeline=hf_llm_pipeline, model_kwargs={'temperature': 0.7})
    temp_document = ""

    # kg_output = kg_chain.run(
    #     {'input_text': """
    # William Henry Gates III (born October 28, 1955) is an American billionaire, philanthropist, and investor best known for co-founding the software giant Microsoft, along with his childhood friend Paul Allen. During his career at Microsoft, Gates held the positions of chairman, chief executive officer (CEO), president, and chief software architect, while also being its largest individual shareholder until May 2014. He was a major entrepreneur of the microcomputer revolution of the 1970s and 1980s.
    # Gates was born and raised in Seattle. In 1975, he and Allen founded Microsoft in Albuquerque, New Mexico. It later became the world's largest personal computer software company. Gates led the company as its chairman and chief executive officer until stepping down as CEO in January 2000, succeeded by Steve Ballmer, but he remained chairman of the board of directors and became chief software architect. During the late 1990s, he was criticized for his business tactics, which were considered anti-competitive. This opinion has been upheld by numerous court rulings. In June 2008, Gates transitioned into a part-time role at Microsoft and full-time work at the Bill & Melinda Gates Foundation, the private charitable foundation he and his then-wife Melinda had established in 2000. He stepped down as chairman of the Microsoft board in February 2014 and assumed the role of technology adviser to support newly appointed CEO Satya Nadella. In March 2020, Gates left his board positions at Microsoft and Berkshire Hathaway to focus on his philanthropic efforts on climate change, global health and development, and education.
    # Since 1987, Gates has been included in the Forbes list of the world's billionaires. From 1995 to 2017, he held the Forbes title of the richest person in the world every year except in 2008 and from 2010 to 2013. In October 2017, he was surpassed by Amazon founder and CEO Jeff Bezos, who had an estimated net worth of US$90.6 billion compared to Gates's net worth of US$89.9 billion at the time. As of September 2023, Gates has an estimated net worth of US$129 billion, making him the fourth-richest person in the world according to Bloomberg Billionaires Index.
    # Later in his career and since leaving day-to-day operations at Microsoft in 2008, Gates has pursued other business and philanthropic endeavors. He is the founder and chairman of several companies, including BEN, Cascade Investment, TerraPower, bgC3, and Breakthrough Energy. He has donated sizable amounts of money to various charitable organizations and scientific research programs through the Bill & Melinda Gates Foundation, reported to be the world's largest private charity. Through the foundation, he led an early 21st century vaccination campaign that significantly contributed to the eradication of the wild poliovirus in Africa. In 2010, Gates and Warren Buffett founded The Giving Pledge, whereby they and other billionaires pledge to give at least half of their wealth to philanthropy.
    # """}
    # )
    #
    # print(kg_output)
    #
    # # retrival_fn = construct_retrival_fn()
    # retrival_qa_chain = construct_retrival_qa_chain(kg_output)
    # print(retrival_qa_chain.run("Please explain about the company related to Bill Gates."))

    def kg_fn(document, question):
        kg_output = kg_chain.run({'input_text': document})
        retrival_qa_chain = construct_retrival_qa_chain(kg_output, llm_pipeline)
        answer = retrival_qa_chain.run(question)
        print(type(kg_output))
        return kg_output, answer

    demo = gr.Interface(
        fn=kg_fn,
        inputs=["text", "text"],
        outputs=["text", "text"],
    )
    demo.launch(share=True)