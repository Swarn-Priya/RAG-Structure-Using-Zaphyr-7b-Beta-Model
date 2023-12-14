# from langchain import PromptTemplate
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import os
from io import BytesIO
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
import gradio as gr

local_llm = "zephyr-7b-beta.Q5_0.gguf"

config = {
'max_new_tokens':1024,
'repetition_penalty':1.1,
'temperature':0.1,
'top_k':50,
'top_p':0.9,
'stream':True,
'threads':int(os.cpu_count() /2), #adjust your CPU
}

llm = CTransformers(
    model=local_llm,
    model_type='mistral',
    lib="avx2", #for CPU  use
    **config
)

prompt_template = """Use the following peices of information to answer the user' question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

model_name = "BAAI/bge-large-en"  #embedding models
model_kwargs = {'device':'cpu'}
encode_kwargs = {'normalize_embeddings':False}

embeddings = HuggingFaceBgeEmbeddings(model_name=model_name,
                                      model_kwargs=model_kwargs,
                                      encode_kwargs=encode_kwargs)

prompt = PromptTemplate(template=prompt_template, input_variables =['context','question'])
load_vector_store =Chroma(persist_directory = "stores/TSLA_cosine",embedding_function = embeddings)
retriever = load_vector_store.as_retriever(search_kwargs={"k":1})
# query = "how much Tesla’s shares are up YoY?"
# semantic_search = retriever.get_relevant_documents(query)
# print(semantic_search)

# print("##############################################")

# chain_type_kwargs = {"prompt":prompt}

# qa = RetrievalQA.from_chain_type(
#    llm = llm,
#    chain_type="stuff",
#    retriever = retriever,
#    return_source_documents=True,
#    chain_type_kwargs = chain_type_kwargs,
#    verbose = True
# )

# response = qa(query)

# print(response)


sample_prompts = ["Is Tesla’s $1T valuation justified?",
"Which company is world's largest car manufacturer?"]

def get_response(input):
    query = input
    chain_type_kwargs = {'prompt':prompt}
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever = retriever,
        return_source_documents = True,
        chain_type_kwargs = chain_type_kwargs,
        verbose = True
        )

    response = qa(query)
    return response


input = gr.Text(
    label = "Prompt",
    show_label = False,
    max_lines = 1,
    placeholder = "Enter your query ",
    container = False
)

iface = gr.Interface(
    fn = get_response,
    inputs = input,
    outputs = "text",
    title ="Financial Documents QnA RAG Implementation using Zephyr 7b",
    decription = "RAG Demo for Zephyr 7b Beta LLM",
    examples = sample_prompts,
    allow_flagging = False,
    allow_screenshot = False
)

iface.launch()