from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate



load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# --------- Step 1(indexing) ---------
video_id = "aZWZXzyYVZI"

try:
    transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
    for snippet in transcript_list:
        print(snippet.text)
    

except TranscriptsDisabled:
    print("No caption available for this video")
    
    
# --------- Step 1.1(text splitting) ----------
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([snippet])


# --------- Step 1.2(generate and store(for now I'm just storing it to a variable) embedding) ------------
vector_store = FAISS.from_documents(chunks, embedding=embedding_model)


# ---------- step2(retriever) ---------
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":4})



# ---------- step 3(augmentation, basically the prompt creation with query and relevant document)
prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
    Only answer based on the provided transcript context.
    If the context is insufficient, just say I don't know.
    
    {context}
    Question: {question}
    """,
    input_variables=["context", "question"]
)

question = "Your desired question"
retrieved_docs = retriever.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)


final_prompt = prompt.invoke({"context":context_text, "question": question})

# -----------step 4 -----------
answer = llm.invoke(final_prompt)
print(answer.content)






# # ---------forming a chain(this is basically same thing from step 3 and 4 but it's more clean, so when running comment out one) ----------
# from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
# from langchain_core.output_parsers import StrOutputParser

# parser = StrOutputParser()

# def format_doc(retrieved_docs):
#     context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
#     return context_text

# parrallel_chain= RunnableParallel({
#     "context": retriever | RunnableLambda(format_doc),
#     "question": RunnablePassthrough()
# })

# main_chain = parrallel_chain | prompt | llm | parser
# main_chain.invoke("Your desired question....")
