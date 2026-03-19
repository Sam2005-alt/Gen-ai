import os
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader

class RAGSystem:
    def __init__(self, document_paths):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = self.setup_vector_store(document_paths)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.embeddings,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever()
        )

    def setup_vector_store(self, document_paths):
        texts = []
        for path in document_paths:
            loader = TextLoader(path)
            texts.extend(loader.load())
        return Chroma.from_documents(texts, self.embeddings)

    def query(self, question):
        return self.qa_chain.run(question)

if __name__ == '__main__':
    document_paths = ["path/to/document1.txt", "path/to/document2.txt"]  # Change to actual document paths
    rag_system = RAGSystem(document_paths)
    question = "What is the purpose of the project?"
    answer = rag_system.query(question)
    print(answer)