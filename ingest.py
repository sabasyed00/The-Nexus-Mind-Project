#Import modules
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
#If the data source is unstructured (PPT) use Unstructed loader
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter 
#Any errors with Hugggingfaceembeddings use sentence transformers instead 
#This happens due to version conflicts

# Defining a data path 
DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'



# Create vector database

#Function to create vector database
###Vector database -  a data structure that stores vectors derived from textual documents in a structured and searchable format
### vectors - used to represent various linguistic elements such as words, sentences, or documents. 
###vectors -  capture the semantic meaning of the text in a numerical format, allowing for fast and efficient similarity search.
##Each dimension in the vector corresponds to a specific feature or property of the object being represented.
def create_vector_db():
    #Initialize the document loader to load PDF documents from the specified path
    loader = DirectoryLoader(DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader) 
    #Load the documents

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)   #Initialize the text splitter to spilt the documents into chunks
    texts = text_splitter.split_documents(documents)  #Split the documents into text chunks

   #Creating embeddings
   #Initialize the HuggingFace embeddings with a specific model
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
     #Create the FAISS vector database from the text chunks and embeddings
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH) #Save the vector database locally      

        #Check if the script is run as the main module
if __name__ == "__main__":
    create_vector_db() #Call the function to create the vector database

##FAISS stands for "Facebook AI Similarity Search," and it's a library for efficient similarity search and clustering of dense vectors. It's particularly well-suited for scenarios where you have a large dataset of high-dimensional vectors and need to quickly find similar vectors.
###FAISS is commonly used in natural language processing (NLP) tasks, such as:
###1. **Information Retrieval**: Searching for similar documents or texts.
###2. **Recommendation Systems**: Finding similar items or products based on user preferences.
###3. **Clustering**: Grouping similar documents or texts together.
###One of the key features of FAISS is its ability to perform fast similarity searches using methods like the inverted file structure, which enables efficient indexing and retrieval of nearest neighbors.
###In the context of the code you provided, FAISS is used to create a vector database from text data, allowing for fast and efficient similarity search operations on that data.