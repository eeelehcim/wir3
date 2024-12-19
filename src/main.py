import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import torch
from sklearn.cluster import KMeans
import pickle
from concurrent.futures import ThreadPoolExecutor

# Paths
prefix = "data/large/" 
full_docs_path = f"{prefix}full_docs/"
queries_path = f"{prefix}queries_test.csv"
results_path = f"{prefix}results.csv"
embeddings_and_ids_path = f"{prefix}embeddings_and_ids.npz"
cluster_data_path = f"{prefix}cluster_data.pkl"

def preprocess_text_chunks(text, max_tokens=512):
    """Split text into chunks of up to max_tokens."""
    tokens = text.split()
    chunks = [" ".join(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)]
    return chunks

def load_model_with_device(model_name="all-MiniLM-L6-v2", device="cuda"):
    """Load SBERT model on the specified device."""
    print(f"Loading model '{model_name}' on device '{device}'...")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Falling back to CPU.")
        device = "cpu"
    model = SentenceTransformer(model_name, device=device)
    print("Model successfully loaded.")
    return model

def load_embeddings_and_ids():
    """Load embeddings and document IDs if the file exists."""
    if os.path.exists(embeddings_and_ids_path):
        print("Loading embeddings and document IDs...")
        data = np.load(embeddings_and_ids_path)
        embeddings = data["embeddings"]
        ids = data["ids"]
        return embeddings, ids
    else:
        print(f"Embeddings file not found. Proceeding with vectorization...")
        return None, None

def process_single_file(file_name, model):
    """Helper function to process a single file and return its embedding and ID."""
    file_path = os.path.join(full_docs_path, file_name)
    try:
        doc_id = int(file_name.split('_')[1].split('.')[0])
        with open(file_path, 'r', encoding='utf-8') as file:
            doc_text = file.read().strip()

            if not doc_text:
                print(f"Document {file_name} is empty. Skipping.")
                return None

            chunks = preprocess_text_chunks(doc_text)
            if not chunks:
                print(f"No valid chunks for document {file_name}. Skipping.")
                return None
            chunk_embeddings = [model.encode(chunk, convert_to_numpy=True) for chunk in chunks]
            document_embedding = np.mean(chunk_embeddings, axis=0)
            return document_embedding, doc_id
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")
        return None

from concurrent.futures import ThreadPoolExecutor, as_completed

def vectorize_documents(model, max_workers=32):
    """Embed all documents in the dataset with multithreading."""
    print("Starting document vectorization...")
    doc_embeddings, doc_ids = load_embeddings_and_ids()
    if doc_embeddings is not None and doc_ids is not None:
        print("Embeddings already exist. Skipping re-computation.")
        return doc_embeddings, doc_ids

    files = [f for f in os.listdir(full_docs_path) if f.startswith("output_") and f.endswith(".txt")]
    total_files = len(files)
    if total_files == 0:
        print("No documents found.")
        return {}, {}
    print(f"Processing {total_files} files with multithreading...")
    doc_embeddings = []
    doc_ids = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_single_file, file_name, model): file_name for file_name in files}

        for i, future in enumerate(as_completed(future_to_file), start=1):
            try:
                result = future.result()
                if result is not None:
                    embedding, doc_id = result
                    doc_embeddings.append(embedding)
                    doc_ids.append(doc_id)
                print(f"Progress: {i}/{total_files} files processed ({(i / total_files) * 100:.2f}%).")
            except Exception as e:
                print(f"Error processing a file: {e}")

    # Save embeddings and IDs
    if doc_embeddings:
        print("Saving document embeddings...")
        doc_embeddings = np.stack(doc_embeddings, axis=0).astype(np.float32)
        doc_ids = np.array(doc_ids, dtype=np.int32)
        np.savez_compressed(embeddings_and_ids_path, embeddings=doc_embeddings, ids=doc_ids)
        print(f"Document embeddings saved to {embeddings_and_ids_path}.")
    else:
        print("No valid documents to save.")

    return doc_embeddings, doc_ids



def load_clusters():
    """Load clustering data if it exists."""
    if os.path.exists(cluster_data_path):
        print("Loading clustering data...")
        with open(cluster_data_path, 'rb') as file:
            cluster_data = pickle.load(file)
        return cluster_data["cluster_centroids"], cluster_data["cluster_assignments"]
    else:
        print("Clustering data not found. Proceeding with clustering...")
        return None, None

def cluster_documents(doc_embeddings, n_clusters=100):
    """Cluster document embeddings using KMeans."""
    cluster_centroids, cluster_assignments = load_clusters()
    if cluster_centroids is not None and cluster_assignments is not None:
        print("Clustering already exists. Skipping re-computation.")
        return cluster_centroids, cluster_assignments

    print("Clustering document embeddings...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_assignments = kmeans.fit_predict(doc_embeddings)
    cluster_centroids = kmeans.cluster_centers_
    print(f"Saving clustering data to {cluster_data_path}...")
    with open(cluster_data_path, 'wb') as file:
        pickle.dump({
            "cluster_centroids": cluster_centroids,
            "cluster_assignments": cluster_assignments
        }, file)
    print("Clustering data saved.")
    return cluster_centroids, cluster_assignments


def retrieve_documents_with_clusters(model, cluster_centroids, cluster_assignments, doc_embeddings, doc_ids, top_k=10):
    """Retrieve top-k documents for each query using cluster-based retrieval."""
    print("Loading queries...")
    query_df = pd.read_csv(queries_path)
    results = []

    for i, (query_number, query) in enumerate(zip(query_df['Query number'], query_df['Query']), start=1):
        print(f"Processing query {i}/{len(query_df)}: '{query}'")
        try:
            query_embedding = model.encode(query, convert_to_numpy=True)

            centroid_similarities = cosine_similarity(query_embedding.reshape(1, -1), cluster_centroids).flatten()
            top_clusters = np.argsort(centroid_similarities)[::-1][:top_k]
            candidate_docs = []
            candidate_doc_ids = []
            for cluster in top_clusters:
                cluster_indices = np.where(cluster_assignments == cluster)[0]
                candidate_docs.extend(doc_embeddings[cluster_indices])
                candidate_doc_ids.extend(doc_ids[cluster_indices])
            candidate_similarities = cosine_similarity(query_embedding.reshape(1, -1), np.stack(candidate_docs)).flatten()
            ranked_indices = np.argsort(candidate_similarities)[::-1][:top_k]
            for rank, idx in enumerate(ranked_indices):
                results.append({
                    "Query_number": query_number,
                    "doc_number": candidate_doc_ids[idx]
                })
        except Exception as e:
            print(f"Error processing query {query_number}: {e}")

    print("Saving results...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}.")

def evaluate_results():
    """Evaluate retrieval results using MP@k and MR@k."""
    print("Evaluating results...")
    relevant_docs_df = pd.read_csv(f"{prefix}relevant_docs.csv")
    results_df = pd.read_csv(results_path)

    processed_query_ids = results_df['Query id'].unique()
    relevant_docs_df = relevant_docs_df[relevant_docs_df['Query_number'].isin(processed_query_ids)]
    relevant_docs = relevant_docs_df.groupby('Query_number')['doc_number'].apply(set).to_dict()
    retrieved_docs = results_df.groupby('Query id')['Document id'].apply(list).to_dict()

    k_values = [1, 3, 5, 10]
    metrics = {k: {"MPk": 0, "MRk": 0} for k in k_values}
    num_queries = len(relevant_docs)

    for query_id in relevant_docs:
        if query_id not in retrieved_docs:
            continue
        relevant_set = relevant_docs[query_id]
        retrieved_list = retrieved_docs[query_id]
        num_relevant = len(relevant_set)

        for k in k_values:
            top_k_retrieved = set(retrieved_list[:k])
            relevant_and_retrieved = relevant_set & top_k_retrieved
            precision_at_k = len(relevant_and_retrieved) / k
            recall_at_k = len(relevant_and_retrieved) / num_relevant if num_relevant > 0 else 0
            metrics[k]["MPk"] += precision_at_k
            metrics[k]["MRk"] += recall_at_k

    for k in k_values:
        metrics[k]["MPk"] /= num_queries
        metrics[k]["MRk"] /= num_queries
        print(f"K={k}: MP@k={metrics[k]['MPk']:.4f}, MR@k={metrics[k]['MRk']:.4f}")

def main():
    print("Semantic Search and Retrieval System")
    model = load_model_with_device()

    print("Step 1: Vectorize documents...")
    doc_embeddings, doc_ids = vectorize_documents(model)

    print("Step 2: Cluster document embeddings...")
    cluster_centroids, cluster_assignments = cluster_documents(doc_embeddings, n_clusters=100)

    print("Step 3: Retrieve documents using clusters...")
    retrieve_documents_with_clusters(model, cluster_centroids, cluster_assignments, doc_embeddings, doc_ids)

    print("Step 4: Evaluate retrieval results...")
    #evaluate_results()

if __name__ == "__main__":
    main()