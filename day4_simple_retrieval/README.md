Simple Semantic Retrieval System (Core of RAG)

This project implements a basic semantic search system using sentence embeddings and cosine similarity. It demonstrates the core idea behind Retrieval-Augmented Generation (RAG) systems.

Overview
Traditional keyword search looks for exact matches, but semantic retrieval understands the meaning of a query.

This system:
Converts documents into vector embeddings
Converts user queries into embeddings
Finds the most semantically similar documents using cosine similarity

📁 Project Structure
├── documents.txt        # Input knowledge base (one document per line)
├── queries.txt          # Sample queries
├── results.txt          # Output results (Top-K matches)
├── semantic_prompt.py   # Main implementation
└── README.md            # Project documentation

⚙️ How It Works
1. Load Documents
Reads a text file where each line is treated as a separate document.

2. Generate Embeddings
Uses a pre-trained transformer model: "all-MiniLM-L6-v2" to convert text into dense vector representations.

3. Query Processing
User enters a query -> Query is converted into an embedding -> Cosine similarity is computed with all document embeddings

4. Retrieve Top-K Results
Returns the most relevant documents ranked by similarity score.


🛠️ Installation
1. Clone the repository
git clone <your-repo-link>
cd <repo-folder>
2. Install dependencies
pip install numpy scikit-learn sentence-transformers

▶️ Usage
Run the script:
python semantic_prompt.py
Then enter queries interactively,
Enter your search query,
Type exit to quit.

🧩 Core Concepts Used
Sentence Embeddings (Transformers)
Cosine Similarity
Semantic Search
Vector Space Representation

📌 Key Learning Outcome
This project demonstrates the foundation of modern AI retrieval systems, including:
ChatGPT-style retrieval pipelines
Document search engines
Knowledge-based QA systems