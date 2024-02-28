#!/usr/bin/env python
# coding: utf-8

# In[1]:


from langchain.llms import OpenAI
import os
os.environ["OPENAI_API_KEY"] = "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"


# In[14]:


from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("0131047023-HDFC-Life-Click2Protect-Elite-Brochure.pdf")
                    
pages = loader.load_and_split()


# In[15]:


pages


# In[16]:


from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(separator = '\n',
                                      chunk_size=1000,
                                      chunk_overlap=200)

docs = text_splitter.split_documents(pages)


# In[17]:


len(docs)


# In[22]:


import pickle
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings
model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
model = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction="为这个句子生成表示以用于检索相关文章："
)
model.query_instruction = "为这个句子生成表示以用于检索相关文章："


# In[23]:


model


# In[24]:


vectorStore_openai = FAISS.from_documents(docs, model)
with open("faiss_store_openai.pkl","wb") as f:
    pickle.dump(vectorStore_openai,f)
with open("faiss_store_openai.pkl","rb") as f:
    vectorStore = pickle.load(f)


# In[25]:


from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
llm=OpenAI(temperature=0,)


# In[26]:


llm


# In[27]:


chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorStore.as_retriever())


# In[28]:


chain({"question":"what are the benefits of payable under the plan?"})


# In[ ]:




