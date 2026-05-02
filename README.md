# Information Retrieval (IR) Model Benchmarking: Cranfield Collection
Section 1:
#Group members:
1. Abiy Ketema UGR/4680/17
2. Abubaker Mohammed UGR/4856/17
3. Eyuel gezahegn UGR/1200/17 
4. Henok Zegeye UGR/5413/17
5. Abdella
6. Arsema Teshome

This project focuses on evaluating various Information Retrieval (IR) architectures using the **Cranfield dataset**. The goal was to track how retrieval accuracy evolves when moving from simple keyword matching to modern neural "Cross-Encoder" re-ranking.

---

##  Table of Contents
* [The Dataset](#the-dataset)
* [Implementation Logic](#implementation-logic)
* [Models Tested](#models-tested)
* [Results & Evaluation](#results)

---

##  The Dataset
We used the **Cranfield Collection**, a classic benchmark in IR consisting of:
* **1,400 Documents**: Abstracts focused on aerodynamics.
* **Standardized Queries**: Fixed natural language questions used to test each model.
* **Ground Truth (Qrels)**: The "gold standard" indicating which documents are actually relevant to each query.

##  Models Tested

### 1. Keyword-Based (Traditional)
* **Boolean (Term Count)**: A baseline approach that ranks documents based on how many query tokens they contain.
* **TF-IDF (Vector Space)**: Ranks documents based on term frequency and statistical rarity across the corpus.
* **BM25**: A more robust probabilistic ranking function often used as the default in production systems like Elasticsearch.

### 2. Neural & Semantic (Modern)
* **Bi-Encoder (Dense Retrieval)**: Uses the `all-MiniLM-L6-v2` transformer to map queries and documents into a vector space. This allows the system to find results based on **context and meaning** rather than just exact word matches.
* **Cross-Encoder (Re-ranker)**: A high-precision pipeline that takes the top 100 results from BM25 and re-evaluates them using a BERT-based model (`ms-marco-MiniLM-L-6-v2`) to determine the final order.

##  Results
The models were evaluated at **Top-10 (K=10)**. Below is the performance breakdown:

| Model | Precision | Recall | F1 | Fallout |
| :--- | :--- | :--- | :--- | :--- |
| **Boolean** | 0.2000 | 0.2714 | 0.1796 | 0.00576 |
| **TF-IDF** | 0.3800 | 0.5190 | 0.3568 | 0.00447 |
| **BM25** | 0.3400 | 0.3940 | 0.3013 | 0.00475 |
| **Bi-Encoder** | **0.4000** | **0.5595** | **0.3883** | **0.00432** |
| **Cross-Encoder** | 0.3600 | 0.5190 | 0.3410 | 0.00461 |

### Key Takeaways
* **Semantic Power**: The **Bi-Encoder** outperformed the other models in this specific test, showing that understanding the "intent" behind a query is often more effective than matching specific technical terms in the Cranfield abstracts.
* **Efficiency vs. Accuracy**: While the Cross-Encoder is technically more sophisticated, its performance in a two-stage pipeline depends heavily on the initial "retrieval" stage.

## Tech Stack
* **Processing**: `scikit-learn`, `rank_bm25`
* **Neural Models**: `sentence-transformers`, `PyTorch`
* **Evaluation**: `pytrec_eval`, `ir_datasets`