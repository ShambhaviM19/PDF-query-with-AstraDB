from langchain_community.vectorstores import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_openai import OpenAI, OpenAIEmbeddings
from datasets import load_dataset
import cassio
from PyPDF2 import PdfReader
from typing_extensions import Concatenate
from langchain.text_splitter import CharacterTextSplitter

import os 
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
astra_db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
astra_db_id = os.getenv("ASTRA_DB_ID")

pdfreader = PdfReader('speech.pdf')

raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

cassio.init(token=astra_db_application_token, database_id=astra_db_id)

llm = OpenAI(openai_api_key=openai_api_key)
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="qa_mini_demo",
    session=None,
    keyspace=None,
)


text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)


astra_vector_store.add_texts(texts[:50])

astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
first_question = True
while True:
    if first_question:
        query_text = input("\nEnter your question (or type 'quit' to exit): ").strip()
    else:
        query_text = input("\nWhat's your next question (or type 'quit' to exit): ").strip()

    if query_text.lower() == "quit":
        break

    if query_text == "":
        continue

    first_question = False

    print("\nQUESTION: \"%s\"" % query_text)
    answer = astra_vector_index.query(query_text, llm=llm).strip()
    print("ANSWER: \"%s\"\n" % answer)

    print("FIRST DOCUMENTS BY RELEVANCE:")
    for doc, score in astra_vector_store.similarity_search_with_score(query_text, k=4):
        print("    [%0.4f] \"%s ...\"" % (score, doc.page_content[:84]))
