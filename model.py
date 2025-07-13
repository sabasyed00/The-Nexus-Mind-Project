#Import modules
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA

import chainlit as cl

import os 

DB_FAISS_PATH = 'vectorstore/db_faiss'  # Define the path for FAISS vector store

# Create an embedding function
embedding_function = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})

# Load the persisted embeddings with the embedding function
def load_faiss_db():
    return FAISS.load_local(DB_FAISS_PATH, embeddings=embedding_function, allow_dangerous_deserialization=True)

# Define a custom prompt template for QA retrieval
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    return PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

# Make sure FAISS is used in the retrieval QA chain
def retrieval_qa_chain(llm, prompt, db):
    return RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff', #stuff all retrieved data
                                       retriever=db.as_retriever(search_kwargs={'k': 2}), #top 2 searches
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )

def answer_question(question):
    # Initialize the QA bot and set up the QA chain
    llm = load_llm()
    prompt = set_custom_prompt()
    db = load_faiss_db()  # Load the FAISS database
    qa_chain = retrieval_qa_chain(llm, prompt, db)
    
    # Find relevant documents
    relevant_docs = db.similarity_search(question)

    # Get the answer using the question answering chain
    answer = qa_chain.run(
        input_documents=relevant_docs,
        question=question
    )
    return answer  

# Example usage
if __name__ == "__main__":
    question = "What is the purpose of this project?"
    answer = answer_question(question)
    print(answer)

# Ensure LLM is loaded correctly
def load_llm():
    # Load the locally downloaded model here
    return CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5 # controls creativity/ randomness of the model 
    )

# QA Model Function that uses FAISS
def qa_bot(allow_dangerous_deserialization=True):
    print("Initializing QA bot....")
    db = load_faiss_db()  # Load the FAISS vector store
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    print("QA bot initialized successfully.")
    return qa

# Handle the final result using the FAISS-based QA bot
def final_result(query):
    qa_result = qa_bot(allow_dangerous_deserialization=True)  # Enable dangerous deserialization
    response = qa_result({'query': query})
    return response

## Chainlit setup ##
@cl.on_chat_start
async def start():
    print("Starting the chat...")
    chain = qa_bot(allow_dangerous_deserialization=True)  # Enable deserialization
    print("QA bot loaded successfully.")
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to NexusMind 1.0. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)
    print("Chain set successfully")

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()