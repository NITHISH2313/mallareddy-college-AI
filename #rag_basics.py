#rag_basics
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough

# ===== YOUR DOCUMENTS =====
documents = [
    """Mallareddy Engineering College is located in Hyderabad, Telangana.
    The college offers BTech programs in CSE, ECE, EEE, Mechanical and Civil.
    The college was established in 2001 and is affiliated to JNTUH.
    Total student strength is approximately 5000 students.""",

    """Exam Rules at Mallareddy Engineering College:
    Students must have 75% attendance to appear for exams.
    Internal exams are conducted every month.
    External exams are conducted at end of semester.
    Students caught cheating will be expelled immediately.
    Results are announced within 30 days after exams.""",

    """Fee Structure at Mallareddy Engineering College:
    BTech CSE fee is 85000 per year.
    BTech ECE fee is 80000 per year.
    Hostel fee is 60000 per year.
    Bus fee ranges from 15000 to 25000 per year.
    Scholarships available for meritorious students.""",

    """Placement Statistics at Mallareddy Engineering College:
    Average placement package is 4.5 LPA.
    Highest package offered is 18 LPA.
    Top recruiting companies include TCS, Infosys, Wipro, Cognizant.
    Placement rate is approximately 75% every year.
    Google and Microsoft visit campus occasionally.""",

    """Nithish Kumar is a BTech CSE student at Mallareddy Engineering College.
    He is 22 years old and from Hyderabad.
    He is learning AIML and wants to become a GenAI engineer.
    His target salary is 15 LPA and dream company is Google.
    He has been learning for 24 days consistently."""
]

# ===== SPLIT DOCUMENTS =====
print("splitting documents...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

chunks = splitter.create_documents(documents)
print(f"total chunks: {len(chunks)}")

# ===== CREATE EMBEDDINGS =====
print("\ncreating embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ===== STORE IN VECTOR DB =====
print("storing in ChromaDB...")
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("vector database created! ✅")

# ===== SETUP RETRIEVER =====
retriever = vectordb.as_retriever(
    search_kwargs={"k": 3}
)

# ===== SETUP LLM =====
api_key = "GROQ_API_KEY"  # paste your key!
llm = ChatGroq(
    api_key=api_key,
    model="llama-3.1-8b-instant",
    temperature=0
)

# ===== RAG PROMPT =====
template = """You are a helpful assistant for 
Mallareddy Engineering College.
Answer the question based on the context below.
If you don't know the answer, say "I don't know".

Context: {context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

# ===== RAG CHAIN =====
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs,
     "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ===== ASK QUESTIONS =====
print("\n===== MALLAREDDY COLLEGE AI =====")
questions = [
    "What is the attendance requirement for exams?",
    "What is the BTech CSE fee?",
    "What is the highest placement package?",
    "Who is Nithish and what is his goal?",
    "Which companies recruit from the college?"
]

for question in questions:
    print(f"\nQ: {question}")
    answer = rag_chain.invoke(question)
    print(f"A: {answer}")
    print("-" * 40)
