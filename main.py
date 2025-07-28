import os
import json
import re
import fitz  # PyMuPDF
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

# Paths (adjust if needed)
INPUT_DIR = "/pdf_extractor/input/pdfs"
OUTPUT_DIR = "/pdf_extractor/output"
MODEL_NAME = "/pdf_extractor/models/all-MiniLM-L6-v2"


# Keywords relevant for HR onboarding & compliance (adjust per persona)
KEYWORDS = [
    "onboarding form", "employee form", "compliance", "HR policy", "acknowledgment",
    "fill out", "employee details", "benefits enrollment", "direct deposit", "contract",
    "EEO", "anti-harassment", "tax form", "employment eligibility", "checklist"
]

# Subqueries to enhance semantic match for HR use case
SUBQUERIES = [
    "fillable onboarding forms",
    "HR compliance checklists",
    "employee data collection forms",
    "policy acknowledgment forms",
    "employment contract and benefits info"
]

def extract_text_by_page(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():
            pages.append((page_num , clean_text(text)))
    return pages

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # normalize whitespace
    text = re.sub(r'-\s+', '', text)  # remove hyphen line breaks
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # remove non-ASCII
    return text.strip()

def contains_keywords(text):
    return any(kw.lower() in text.lower() for kw in KEYWORDS)

def rank_sections(sections, model, queries):
    scored_sections = []

    for query in queries:
        query_embedding = model.encode(query, convert_to_tensor=True)
        texts = [s["text"] for s in sections]
        section_embeddings = model.encode(texts, convert_to_tensor=True)

        similarities = util.pytorch_cos_sim(query_embedding, section_embeddings)[0]

        for i, sim in enumerate(similarities):
            score = float(sim)
            if contains_keywords(sections[i]["text"]):
                score += 0.15  # Boost if keywords are found

            scored_sections.append((sections[i], score))

    # Deduplicate and keep top 5
    seen = set()
    unique_sections = []
    for section, score in sorted(scored_sections, key=lambda x: x[1], reverse=True):
        uid = (section["document"], section["page_number"])
        if uid not in seen:
            section["similarity"] = round(score, 4)
            unique_sections.append(section)
            seen.add(uid)
        if len(unique_sections) == 5:
            break

    return unique_sections

def main():
    model = SentenceTransformer(MODEL_NAME)

    # Load persona & job from persona.json
    with open(os.path.join(INPUT_DIR, "persona.json"), "r") as f:
        persona_data = json.load(f)

    job = persona_data["job"]
    persona = persona_data["persona"]

    # Prepare enhanced queries
    enhanced_queries = SUBQUERIES + [f"{persona}: {job}"]

    # Load and extract from all PDFs
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".pdf")]
    sections = []

    for file in files:
        pdf_path = os.path.join(INPUT_DIR, file)
        pages = extract_text_by_page(pdf_path)
        for page_num, text in pages:
            snippet = text[:1200]  # limit to 1200 chars
            sections.append({
                "document": file,
                "page_number": page_num,
                "section_title": text.split('\n')[0][:120],
                "text": snippet
            })

    # Rank and filter most relevant sections
    top_sections = rank_sections(sections, model, enhanced_queries)

    # Build metadata
    metadata = {
        "input_documents": files,
        "persona": persona,
        "job_to_be_done": job,
        "processing_timestamp": datetime.utcnow().isoformat()
    }

    extracted_sections = [
        {
            "document": s["document"],
            "section_title": s["section_title"],
            "importance_rank": i + 1,
            "page_number": s["page_number"]
        }
        for i, s in enumerate(top_sections)
    ]

    subsection_analysis = [
        {
            "document": s["document"],
            "refined_text": s["text"],
            "page_number": s["page_number"]
        }
        for s in top_sections
    ]

    output_data = {
        "metadata": metadata,
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }

    # Save result
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "output.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("✅ Extraction complete. Output saved to:", output_path)

if __name__ == "__main__":
    main()





