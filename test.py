from src.utils import get_embedding_model
from src.retriever import load_vectorstore

em = get_embedding_model()
vs = load_vectorstore(em)
data = vs.get()
print(vs._collection.name)
print(data.keys())
print(len(data['documents']))
print(data['documents'][0][:200])