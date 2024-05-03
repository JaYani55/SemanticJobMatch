import openai
import chromadb
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

# Initialize the ChromaDB client and create a collection
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="test_job_collection")

# Metadata cache
metadata_cache = {}

# Add documents with metadata to the collection
documents = [
    "Senior Software Engineer needed with experience in Python, JavaScript, and cloud services.",
    "Digital Marketing Specialist required with expertise in SEO, PPC campaigns, and social media management.",
    "Umsatzverantwortung für ein eigenes B2B-Kund:innennportfolio, Entwicklung eines tiefen Verständnisses für die Bedürfnisse der B2B-Kund:innen, Verkauf passender Lösungen aus dem Produktportfolio, Cross- & Upselling-Maßnahmen, Unterstützung des Kund:innenportfolios durch individuelle Kampagnen und Performance Management, Erreichung von Umsatzzielen und KPIs",
    "Mitwirkung bei der Erstellung von Monats- und Jahresabschlüssen, Mitarbeit bei Budgetierung und Forecasting-Prozessen, Übernahme des Investitionscontrollings und monatliches Investitions-Reporting",
    "Informieren und Beraten von Stores bei IT-Problemen, Entwickeln und Betreuen von IT-Lösungen, Durchführen von qualitätssichernden Maßnahmen",
    "Der Digital Marketing Assistent unterstützt das E-Commerce-Team bei Ad-hoc-Projekten und übernimmt die Produktpflege auf den digitalen Kanälen der Firma. Er/Sie arbeitet mit verschiedenen Social Media Kanälen und aktuellen Trends und hat eine hohe Lernbereitschaft."
]
metadatas = [
    {"title": "Senior Software Engineer", "skills": ", ".join(["Python", "JavaScript", "cloud services"])},
    {"title": "Digital Marketing Specialist", "skills": ", ".join(["SEO", "PPC campaigns", "social media management"])},
    {"title": "(Junior) Strategic Account Manager (d/w/m)", "skills": ", ".join(["B2B", "Kundenportfolio", "Verkauf", "Cross- & Upselling", "Kampagnen", "Performance Management", "Umsatzzielen", "KPIs"])},
    {"title": "Junior Controller (w/m/d)", "skills": ", ".join(["Monats- und Jahresabschlüssen", "Budgetierung", "Forecasting-Prozessen", "Investitionscontrolling", "Investitions-Reporting"])},
    {"title": "Werkstudent (m/w/d) im Digital Marketing", "skills": ", ".join(["Informieren und Beraten von Stores bei IT-Problemen", "Entwickeln und Betreuen von IT-Lösungen", "Durchführen von qualitätssichernden Maßnahmen"])}
]
ids = ["job_post_1", "job_post_2", "job_post_3", "job_post_4", "job_post_5"]

for doc, meta, id_ in zip(documents, metadatas, ids):
    collection.add(documents=[doc], metadatas=[meta], ids=[id_])
    metadata_cache[id_] = meta

# Query the collection
results = collection.query(
    query_texts=[
        "Ich bin Marketing Spezialist und suche eine Stelle in der Digitalbranche.",
    ],
    n_results=5
)
print(f"Query results: {results}")

# Parse the query results into a dictionary
results_dict = {
    "ids": results["ids"][0],
    "distances": results["distances"][0]
}

# Normalize the distances to a common scale
max_distance = max(results_dict["distances"])
min_distance = min(results_dict["distances"])
normalized_distances = [(distance - min_distance) / (max_distance - min_distance) for distance in results_dict["distances"]]

# Calculate the similarity score based on the normalized distances
similarity_scores = [(1 - distance) * 100 for distance in normalized_distances]

# Create a JSON dictionary for the results
output = []
for id_, score in zip(results_dict["ids"], similarity_scores):
    output.append({"id": id_, "score": score})

# Print the results
print(json.dumps(output, indent=2))