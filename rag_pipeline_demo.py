from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import EmbeddingRetriever, PromptNode, PromptTemplate
from haystack.pipelines import Pipeline

# 1. Create an in-memory document store and insert sample documents

document_store = InMemoryDocumentStore(use_bm25=False)

docs = [
    {"content": "A savings account is a deposit account held at a bank that earns interest."},
    {"content": "Credit cards allow you to borrow funds up to a pre-approved limit for purchases."},
    {"content": "Mortgage loans are secured loans in which real estate is used as collateral."},
    {"content": "The central bank may adjust interest rates to control inflation."},
]

document_store.write_documents(docs)

# 2. Dense retriever for embedding & retrieving docs
retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
)

document_store.update_embeddings(retriever)

# 3. Define prompt template
prompt_template = PromptTemplate(
    prompt="""
Given these documents:\n{join(documents)}\n\nAnswer the question: {query}
""",
)

# 4. Prompt node using flan-t5-base
prompt_node = PromptNode(
    model_name_or_path="google/flan-t5-base",
    default_prompt_template=prompt_template,
    max_length=128,
)

# 5. Build pipeline
pipeline = Pipeline()
pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
pipeline.add_node(component=prompt_node, name="Generator", inputs=["Retriever"])

if __name__ == "__main__":
    query = "How do savings accounts earn money?"
    result = pipeline.run(query=query, params={"Retriever": {"top_k": 3}})
    for answer in result["results"]:
        print(answer)
