import httpx
import time
import os
from datetime import datetime
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_parse import LlamaParse
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv

llm = llm = HuggingFaceInferenceAPI(
    model_name="meta-llama/Llama-3.2-1B"
)

parser = LlamaParse(api_key=os.getenv("LLAMA_INDEX_API"), result_type='markdown')
file_extractor = {'.pdf': parser}
documents = SimpleDirectoryReader('data/', file_extractor=file_extractor).load_data()

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
query_engine = vector_index.as_query_engine(llm=llm)

def query_with_retry(query, max_retries=3, wait_time=5):
    for attempt in range(max_retries):
        try:
            start_time = datetime.now()
            response = query_engine.query(query)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            print(f"Query completed in {duration:.2f} seconds.\n {response}")
            return response
        except httpx.ReadTimeout:
            if attempt < max_retries - 1:
                print(f"Timeout occurred. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise
        except Exception as e:
            print(f"An error occurred: {e}")
            break

if __name__ == "__main__":
    q3= 'Your task is to act as my personal [UHV] professor. Provide a detailed, well-structured explanation on the topic of [What are the programs needed to achieve the comprehensive human goal?]. Begin with an engaging introduction, followed by a comprehensive description, and break down key concepts under relevant subheadings. The content should be thorough and professionally written, similar to educational resources found on sites like GeeksforGeeks, JavaTpoint, and other learning platforms'
    print(query_with_retry(q3))