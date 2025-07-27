import os
import json
from pathlib import Path
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import List, Dict

# ✅ Environment Key Setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "AIzaSyBDZxhcoyfHxTwNq4-3uJsKeWyLD0seSIY"
if not GEMINI_API_KEY:
    raise ValueError("Missing Gemini API key.")

# ✅ Embedding Wrapper (with Retry)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8))
def embed_safely(embedder: Embeddings, texts: List[str]) -> List[List[float]]:
    return embedder.embed_documents(texts)

# ✅ Gemini Embedding Class
class GeminiEmbeddings(Embeddings):
    def __init__(self, task_type="retrieval_document", output_dim=768):
        self.task_type = task_type
        self.output_dim = output_dim
        self.model_name = "models/text-embedding-004"
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        self.client = genai

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            try:
                res = self.client.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type=self.task_type,
                    output_dimensionality=self.output_dim
                )
                vec = None
                
                # Handle raw list
                if isinstance(res, list):
                    vec = res[0] if isinstance(res[0], list) else res
                
                # Handle raw dict
                elif isinstance(res, dict):
                    if "embedding" in res:
                        embedding = res["embedding"]
                        if isinstance(embedding, dict):
                            vec = embedding.get("values", embedding)
                        else:
                            vec = embedding
                    elif "embeddings" in res:
                        emb = res["embeddings"][0]
                        if isinstance(emb, dict):
                            vec = emb.get("values", emb)
                        else:
                            vec = emb
                
                # Handle Gemini object
                elif hasattr(res, "embedding"):
                    embedding = res.embedding
                    vec = getattr(embedding, "values", embedding)
                elif hasattr(res, "embeddings"):
                    emb = res.embeddings[0]
                    if hasattr(emb, "values"):
                        vec = emb.values
                    else:
                        vec = emb
                
                # Validate and append result
                if vec and isinstance(vec, list) and len(vec) == self.output_dim:
                    results.append(vec)
                else:
                    print(f"⚠️ Invalid vector format or dimension mismatch: expected {self.output_dim}, got {len(vec) if isinstance(vec, list) else type(vec)}")
                    results.append([0.0] * self.output_dim)
                    
            except Exception as e:
                print(f"⚠️ embed_documents failed: {e}")
                results.append([0.0] * self.output_dim)
        
        return results

    def embed_query(self, text: str) -> List[float]:
        try:
            res = self.client.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_query",  # Changed from document to query
                output_dimensionality=self.output_dim
            )
            
            # Handle different response formats
            if hasattr(res, "embedding"):
                embedding = res.embedding
                return embedding.values if hasattr(embedding, "values") else embedding
            elif hasattr(res, "embeddings"):
                emb = res.embeddings[0]
                return emb.values if hasattr(emb, "values") else emb
            elif isinstance(res, dict):
                if "embedding" in res:
                    embedding = res["embedding"]
                    return embedding.get("values", embedding) if isinstance(embedding, dict) else embedding
            elif isinstance(res, list):
                return res[0] if isinstance(res[0], list) else res
            
            print(f"⚠️ Unexpected embedding format: {type(res)}")
            return [0.0] * self.output_dim
            
        except Exception as e:
            print(f"⚠️ embed_query failed: {e}")
            return [0.0] * self.output_dim

# ✅ Document Loader with Strict Metadata Cleaning
def load_documents(folder: str) -> List[Document]:
    files = [f for f in os.listdir(folder) if f.endswith(".json")]
    docs = []
    for fname in files:
        try:
            with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
            entries = data if isinstance(data, list) else [data]
            for entry in entries:
                name = (entry.get("name") or entry.get("sheet_name") or entry.get("company_name") or 
                        entry.get("fund_name") or "Unnamed")
                desc = entry.get("description", entry.get("desc", ""))
                content = f"{name}: {desc}" if desc else name
                
                # Clean metadata - ensure all values are simple types
                def clean_metadata_value(value):
                    """Convert any value to a simple string/int/float/bool/None"""
                    if value is None:
                        return None
                    elif isinstance(value, (str, int, float, bool)):
                        return value
                    elif isinstance(value, list):
                        return ", ".join(str(item) for item in value if item is not None)
                    elif isinstance(value, dict):
                        return str(value)
                    else:
                        return str(value)
                
                raw_metadata = {
                    "source": fname,
                    "name": name,
                    "website": entry.get("website", ""),
                    "linkedin": entry.get("linkedin", ""),
                    "twitter": entry.get("twitter", ""),
                    "crunchbase": entry.get("crunchbase", ""),
                    "focus": entry.get("focus", ""),
                    "sector": entry.get("sector", entry.get("region_or_sector", "")),
                    "stages": entry.get("stages", ["Pre-Seed", "Seed", "Series A"]),
                    "tags": entry.get("tags", [])
                }
                
                # Clean all metadata values
                metadata = {}
                for key, value in raw_metadata.items():
                    cleaned_value = clean_metadata_value(value)
                    if cleaned_value and str(cleaned_value).strip():  # Only add non-empty values
                        metadata[key] = str(cleaned_value).strip()
                
                docs.append(Document(page_content=content, metadata=metadata))
        except Exception as e:
            print(f"❌ Error loading {fname}: {e}")
    print(f"📦 Loaded {len(docs)} VC documents.")
    return docs

# ✅ Load + Embed + Index
print("Loading documents...")
docs = load_documents(r"D:\Desktop\New folder (4)")  # Replace with your folder

print("Filtering complex metadata...")
docs = filter_complex_metadata(docs)

# Double-check metadata cleaning
print("Double-checking metadata...")
for i, doc in enumerate(docs):
    for key, value in doc.metadata.items():
        if not isinstance(value, (str, int, float, bool, type(None))):
            print(f"⚠️ Doc {i}: {key} = {value} ({type(value)})")
            # Force convert to string
            doc.metadata[key] = str(value)

print("Creating embeddings...")
doc_chunks = [doc.page_content for doc in docs]
embedder = GeminiEmbeddings()

# Create vectorstore using from_texts (the correct method)
print("Creating vector store...")
vectorstore = Chroma.from_texts(
    texts=doc_chunks,
    embedding=embedder,  # Pass the embedding object, not pre-computed vectors
    metadatas=[doc.metadata for doc in docs],
    collection_name="vc_chroma"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# ✅ Gemini LLM Setup - Fixed model name
print("Setting up LLM...")
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Fixed: using correct chat model
    google_api_key=GEMINI_API_KEY, 
    temperature=0.3
)

# ✅ Custom Prompt (enforcing relevance-first logic)
custom_prompt = PromptTemplate.from_template("""
You are a VC matchmaker AI. Based ONLY on the following VC information provided in the context, recommend the most relevant VCs for the startup idea.

VC Database Context:
{context}

Startup Idea: "{question}"

Instructions:
1. ONLY use VCs mentioned in the context above
2. For each recommended VC, provide:
   - Name (exactly as shown in context)
   - Sector/Focus (from their profile)
   - Investment stages (from their profile)
   - Website (if available)
   - LinkedIn (if available)  
   - Twitter (if available)
   - Crunchbase (if available)
   - Brief reason why they're relevant to this startup idea

3. Format your response as a numbered list
4. If no relevant VCs are found in the context, say "No specific matches found in the database"
5.For Example This is the expected output format
                                             1. Blume Ventures
   💬 These Bangalore-based VCs are total product enthusiasts who'll geek out with you over your tech stacks.
   🌐 https://www.blume.vc/
   💼 https://www.linkedin.com/company/blume-ventures
   🐦 https://twitter.com/BlumeVentures
   🧠 https://www.crunchbase.com/organization/blume-ventures
   🏷️ Tags: Pre-Seed, Seed, Series A

2. Orios Venture Partners
   💬 This Chennai-based VC firm is on the pulse of Tamil Nadu's startup scene and has a sweet spot for AI and enterprise tech.
   🌐 https://www.orioseventure.com/
   💼 https://www.linkedin.com/company/orios-venture-partners
   🐦 https://twitter.com/OriosVP
   🧠 https://www.crunchbase.com/organization/orios-venture-partners
   🏷️ Tags: Pre-Seed, Seed

3. Venturi Partners
   💬 These Chennai-based VCs have a soft spot for hardware and wearables, and are known to back founders with a solid product-market fit.
   🌐 https://www.venturipartners.in/
   💼 https://www.linkedin.com/company/venturi-partners
   🐦 https://twitter.com/VenturiPartners
   🧠 https://www.crunchbase.com/organization/venturi-partners
   🏷️ Tags: Seed, Series A

4. Indigo Equity Partners
   💬 This Chennai-based VC firm has a strong network of enterprise clients and is always on the lookout for innovative software solutions.
   🌐 https://www.indigoequity.in/
   💼 https://www.linkedin.com/company/indigo-equity-partners
   🐦 https://twitter.com/IndigoEquity
   🧠 https://www.crunchbase.com/organization/indigo-equity-partners
   🏷️ Tags: Series A, Series B

5. Exfinity Ventures
   💬 These Bangalore-based VCs have a keen eye for AI and enterprise tech, and are known to back founders with a strong go-to-market strategy.
   🌐 https://www.exfinity.com/
   💼 https://www.linkedin.com/company/exfinity-ventures
   🐦 https://twitter.com/ExfinityVentures
   🧠 https://www.crunchbase.com/organization/exfinity-ventures
   🏷️ Tags: Pre-Seed, Series A 

Recommended VCs:
""")

# ✅ Final Retrieval Chain
print("Creating QA chain...")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt}
)

# 🔍 Run VC Retrieval
print("Running query...")
query = "We're building a climate tech SaaS for carbon emission tracking across supply chains."

# First, let's test the retriever to see what documents we're getting
print("Testing retriever...")
try:
    relevant_docs = retriever.get_relevant_documents(query)
    print(f"Found {len(relevant_docs)} relevant documents")
    
    # Show first few matches for debugging
    print("\n🔍 Top retrieved documents:")
    for i, doc in enumerate(relevant_docs[:3], 1):
        print(f"{i}. {doc.metadata.get('name', 'Unknown')}")
        print(f"   Content: {doc.page_content[:100]}...")
        print(f"   Sector: {doc.metadata.get('sector', 'N/A')}")
        print(f"   Focus: {doc.metadata.get('focus', 'N/A')}")
        print()
    
    # Now try the QA chain
    print("Running QA chain...")
    response = qa_chain.invoke({"query": query})
    
    # Extract the result from the response
    result = response.get("result", response) if isinstance(response, dict) else response
    
    print("\n🔍 AI Recommended VCs:\n")
    print(result)
    
except Exception as e:
    print(f"❌ Error: {e}")
    
    # Enhanced fallback with better formatting
    print("\n🔍 Fallback - Direct retrieval results:\n")
    try:
        relevant_docs = retriever.get_relevant_documents(query)
        
        for i, doc in enumerate(relevant_docs[:5], 1):
            name = doc.metadata.get('name', 'Unknown')
            sector = doc.metadata.get('sector', 'N/A')
            focus = doc.metadata.get('focus', 'N/A')
            stages = doc.metadata.get('stages', 'N/A')
            website = doc.metadata.get('website', 'N/A')
            linkedin = doc.metadata.get('linkedin', 'N/A')
            twitter = doc.metadata.get('twitter', 'N/A')
            crunchbase = doc.metadata.get('crunchbase', 'N/A')
            
            print(f"{i}. {name}")
            print(f"   🎯 Sector: {sector}")
            print(f"   🔍 Focus: {focus}")
            print(f"   💰 Stages: {stages}")
            if website != 'N/A': print(f"   🌐 Website: {website}")
            if linkedin != 'N/A': print(f"   👥 LinkedIn: {linkedin}")
            if twitter != 'N/A': print(f"   🐦 Twitter: {twitter}")
            if crunchbase != 'N/A': print(f"   📊 Crunchbase: {crunchbase}")
            print(f"   📝 Description: {doc.page_content}")
            print()
            
    except Exception as e2:
        print(f"❌ Fallback also failed: {e2}")

# Additional debugging - check total documents in vectorstore
try:
    print(f"\n📊 Total documents in vectorstore: {vectorstore._collection.count()}")
except:
    pass