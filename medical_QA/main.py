import argparse
import rag_vec

def main(query):

    """
    Calls rag_vec.rag with the given query.
    """

    print(f"Executing RAG with:")
    print(f"  - Query: '{query}'")
    rag_vec.rag(query)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAG process.")
    parser.add_argument("--query", type=str, help="The query string to use.")

    args = parser.parse_args()

    main(args.query)
