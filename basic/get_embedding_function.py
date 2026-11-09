from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

def get_embedding_function():
    return HuggingFaceEmbeddings(
        model_name="models/labse",
        model_kwargs={
            "local_files_only": True,
            "device": "cpu"          
        }
    )

