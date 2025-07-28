# 🧠 Adobe Hackathon 2025 — Round 1B
### Persona-Driven Document Intelligence

This solution implements a **semantic document analysis pipeline** that extracts and ranks the most relevant sections from a collection of PDFs — tailored to a user’s **persona** and their **job-to-be-done**.

It is built for **Round 1B** of Adobe’s “Connecting the Dots” Hackathon and runs entirely **offline**, within CPU and model constraints, with no external API or internet calls.

---

## ✅ What This Solution Does

- Accepts:
  - A **JSON file (`persona.json`)** defining the persona and task
  - A folder of **PDFs (3–10)** as the document collection
- Extracts:
  - The most relevant **sections** and **subsections** across all PDFs using semantic similarity
  - Matches are boosted if **HR-related keywords** are found (can be customized)
- Outputs:
  - A structured `output.json` file with:
    - Metadata
    - Ranked sections
    - Subsection-level content

---

## 🧠 Approach Summary

1. **PDF Preprocessing**
   - Each PDF is parsed page-by-page using `PyMuPDF`
   - Non-ASCII characters and hyphen breaks are cleaned
   - Top snippets (~1200 characters) from each page are extracted

2. **Semantic Scoring**
   - Uses `sentence-transformers` (MiniLM model) to embed:
     - The **persona-task query**
     - A list of **domain-specific subqueries**
     - All extracted sections from all PDFs
   - Cosine similarity is used to match each query to the most relevant text snippets

3. **Boosted Relevance**
   - Sections that contain **HR-related keywords** get a score boost (+0.15) for better alignment in that domain

4. **Result Ranking**
   - Top 5 most relevant sections (by max score across all queries) are retained
   - Duplicate sections (same page/document) are filtered
   - Results are compiled into an output JSON file

---



