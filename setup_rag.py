from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# ۱. بارگذاری فایل متنی مستندات
loader = TextLoader("guide.txt", encoding="utf-8")
docs = loader.load()

# ۲. خرد کردن متن به قطعات کوچک‌تر (Chunking)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# ۳. اتصال به مدل امبدینگ اولاما
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# ۴. ساخت دیتابیس برداری و ذخیره آن در پوشه محلی chroma_db
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print(f"✅ تعداد {len(splits)} قطعه از مستندات با موفقیت پردازش و در دیتابیس ChromaDB ذخیره شد.")