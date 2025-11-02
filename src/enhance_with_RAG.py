from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import os
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

INTERNSHIP_FOLDER = "../data"
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#chromadb başlatma
persist_directory = "./chroma_internship_db"

# Eğer DB zaten varsa yükle yoksa boş oluştur
if os.path.exists(persist_directory):
    chroma_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )
    print("Mevcut Chroma DB yüklendi.")
else:
    chroma_db = Chroma.from_texts(
        texts=["initialization"],  # En az bir text gerekli
        embedding=embedding_function,
        persist_directory=persist_directory
    )
    print("Yeni Chroma DB oluşturuldu.")

# Helper functions
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_html(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
        return soup.get_text()

def load_documents(folder_path):
    docs = []
    metadatas = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        try:
            if file.endswith(".pdf"):
                text = extract_text_from_pdf(file_path)
                docs.append(text)
                metadatas.append({"source": file, "type": "pdf"})
            elif file.endswith(".html"):
                text = extract_text_from_html(file_path)
                docs.append(text)
                metadatas.append({"source": file, "type": "html"})
            elif file.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    docs.append(text)
                    metadatas.append({"source": file, "type": "txt"})
        except Exception as e:
            print(f"Hata ({file}): {e}")
    return docs, metadatas

def add_documents_to_db():
    documents, metadatas = load_documents(INTERNSHIP_FOLDER)
    if documents:
        chroma_db.add_texts(texts=documents, metadatas=metadatas)
        print(f" {len(documents)} belge başarıyla veritabanına eklendi!")
    else:
        print("⚠ Eklenecek belge bulunamadı!")

# RAG için tool
@tool
def search_internship_docs(query: str) -> str:
    """
    Gebze Teknik Üniversitesi staj yönergeleri hakkında bilgi aramak için kullanılır.
    Staj ile ilgili tüm sorular için bu tool kullanılmalıdır.
    """
    retriever = chroma_db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)    

    if not docs:
        return "İlgili bilgi bulunamadı."
    
    context = "\n\n---\n\n".join([f"Kaynak: {doc.metadata.get('source', 'Bilinmiyor')}\n{doc.page_content[:500]}..." 
                                   for doc in docs])
    return f"Staj yönergelerinden ilgili bilgiler:\n\n{context}"

@tool
def search_web(query: str) -> str:
    """
    Web'de arama yapmak için kullanılır. Güncel bilgiler için kullanışlıdır.
    """
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=3):
                results.append(r["body"])
        return "\n\n".join(results) if results else "Sonuç bulunamadı."
    except Exception as e:
        return f"Web araması başarısız: {e}"

@tool
def analyze_data(data: str) -> str:
    """Veri analizi yapar ve içgörüler döndürür."""
    return f"Analiz edilen veri: {data[:200]}..."

@tool
def send_email(recipient: str, message: str) -> str:
    """E-posta gönderir (simülasyon)."""
    return f"✉ E-posta '{recipient}' adresine gönderildi: {message[:50]}..."

# Agent oluşturma
agent = create_agent(
    model=llm,
    tools=[search_internship_docs, search_web, analyze_data, send_email],
    system_prompt=(
        "Sen Gebze Teknik Üniversitesi'nin staj konularında uzman bir asistansın. her şeyi biliyorsun. "
        "search_internship_docs tool'unu MUTLAKA kullan. "
        "Kısa, net ve Türkçe cevaplar ver ve Lütfen başka bir yere yönlendirme."
    )
)

def get_result_text(result):
    all_res = result["messages"][-1].content
    if isinstance(all_res, list):
        res = all_res[0]
        return res.get("text", str(res))
    return str(all_res)

def run():
    print("=" * 50)
    print("GTÜ Staj Asistanı")
    print("=" * 50)
    print("\nDokümanlar yükleniyor...")
    
    add_documents_to_db()

    print("\nHazır! Sorularınızı yazabilirsiniz.")
    print("Çıkmak için 'exit' yazın.\n")
    
    while True: 
        user_content = input(">> ")
        
        if user_content.lower() in ["exit", "çıkış", "quit"]:
            print("\n👋 Görüşmek üzere!")
            break
            
        if not user_content.strip():
            continue
            
        try:
            result = agent.invoke({
                "messages": [
                    {"role": "user", "content": user_content}
                ]
            })
            print("\n" + get_result_text(result) + "\n")
        except Exception as e:
            print(f"\nHata oluştu: {e}\n")

if __name__ == "__main__":
    run()