from doc2skill.ingest.base import Document


def distill_passthrough(documents: list[Document]) -> list[Document]:
    """No-op pass-through for document distillation (placeholder for LLM synthesis)."""
    return [
        Document(
            path=doc.path,
            content=doc.content,
            title=doc.title,
            metadata=dict(doc.metadata),
        )
        for doc in documents
    ]
