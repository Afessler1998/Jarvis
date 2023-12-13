from hashlib import md5
import openai
import os
import pinecone
import dotenv


class Pinecone_Interface:
    def __init__(self):
        dotenv.load_dotenv(".env")
        try:
            pinecone.init(
                api_key=os.getenv("PINECONE_API_KEY"),
                environment=os.getenv("PINECONE_ENVIRONMENT")
            )
            self.index = pinecone.Index(os.getenv("PINECONE_INDEX"))
            self.embedding_model = "text-embedding-ada-002"
            self.client = openai.OpenAI()
        except Exception as e:
            print("ERROR INITIALIZING PINECONE: ", e)

    def vectorize_text(self, text):
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            vec_text_list = [embedding_data.embedding for embedding_data in response.data]
            return vec_text_list
        except Exception as e:
            print("ERROR VECTORIZING TEXT: ", e)
            return []

    def upsert_to_index(self, text):
        try:
            vectors = self.vectorize_text(text)
            if vectors:
                vector_id = str(md5(text.encode()).hexdigest())
                metadata = {'text': text}
                upsert_response = self.index.upsert(
                    vectors=[
                        {
                            'id': vector_id,
                            'values': vectors[0],
                            'metadata': metadata
                        }
                    ]
                )
            else:
                print("Error: Vectorization failed.")
        except Exception as e:
            print("ERROR UPSERTING TO INDEX: ", e)

    def query_index(self, query_text, top_k=5):
        try:
            query_vector = self.vectorize_text(query_text)
            if query_vector:
                query_response = self.index.query(
                    vector=query_vector[0],
                    top_k=top_k, 
                    include_metadata=True,
                )
                returned_text = [match.metadata['text'] for match in query_response['matches']]
                return returned_text
            else:
                print("Error: Query vectorization failed.")
                return []
        except Exception as e:
            print(f"ERROR QUERYING INDEX: {type(e).__name__}, {e}")

