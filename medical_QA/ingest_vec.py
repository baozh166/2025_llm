import pandas as pd
import os, time, json, subprocess
from qdrant_client import QdrantClient, models

def load_index(data_path="../data/medquad.csv", collection_name = "medicalQA-rag1020"):

    # 1. load data as dictionary
    df = pd.read_csv(data_path) # read data
    df = df.dropna() # drop nan
    documents = df.to_dict(orient='records') # convert to dictionary
    documents = documents[:500] # too long to embed
    
    
    # 2. Run Qdrant instance
    # --- Step 2.1: Pull the qdrant image (this should run synchronously) ---
    pull_command = ["docker", "pull", "qdrant/qdrant"]
    try:
        print("Pulling Qdrant Docker image...")
        subprocess.run(pull_command, check=True)
        print("Image pull successful.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to pull Qdrant image: {e}")
        exit(1)
    
    # --- Step 2.2: Run the container in the background using subprocess.Popen ---
    print("Starting Qdrant Docker container in the background...")
    
    run_command = [
        "docker", "run",
        # The -d flag runs the container in 'detached' mode, which is crucial.
        "-d",
        "-p", "6333:6333",
        "-p", "6334:6334",
        # Use os.getcwd() for portability.
        "-v", f"{os.getcwd()}/qdrant_storage:/qdrant/storage:z",
        "qdrant/qdrant"
    ]
    
    # Use Popen instead of run() to start the process without waiting.
    # Set stdout and stderr to DEVNULL to prevent the subprocess from interfering
    # with your main script's output.
    try:
        process = subprocess.Popen(run_command,
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
        print(f"Docker process started with PID: {process.pid}")

        # Add a short delay here to give the container some time to initialize
        print("Waiting 3 seconds for the container to become ready...")
        time.sleep(3)

        # --- Step 2.3: Your subsequent Python code continues here ---
        print("Continuing with the rest of the Python script.")

    except FileNotFoundError:
        print("Error: 'docker' command not found. Please ensure Docker is installed and in your system's PATH.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


    # 3. initialize the client
    client_q = QdrantClient("http://localhost:6333") #connecting to local Qdrant instance


    # 4. creat the collection if not exists
    collection_name = collection_name
    EMBEDDING_DIMENSIONALITY =512
    # select a proper embedding mode with FastEmbed
    model_handle = "jinaai/jina-embeddings-v2-small-en" #embed text locally with model_handle from FastEmbed

    # check if collection exists
    if not client_q.collection_exists(collection_name=collection_name):
        
        # 4.1 Create the collection with specified vector parameters
        client_q.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=EMBEDDING_DIMENSIONALITY,  # Dimensionality of the vectors
                distance=models.Distance.COSINE  # Distance metric for similarity search
            )
        )

  
        # 4.2 define points
        points = []
        id = 0
       
        for doc in documents:
            point = models.PointStruct( # models is an imported module
                id=id,
                vector=models.Document(text=doc['answer'], model=model_handle), 
                payload={
                    "text": doc['answer'], # have to be here or else not vectorized.
                    "source": doc['source'],
                    "focus_area": doc['focus_area']
                } #save all needed metadata fields
            )
            points.append(point)
        
            id += 1
        
        
        # 4.3. upload points to Qdrant
        client_q.upsert(
            collection_name=collection_name,
            points=points
        )

    return client_q, model_handle
