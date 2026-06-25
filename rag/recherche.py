"""
Exercice 0.2 — Recherche sémantique par bi-encoder.

Le bi-encoder encode la requête et interroge ChromaDB pour trouver les top-k
documents les plus proches dans l'espace des embeddings.

Note sur les distances cosinus dans ChromaDB :
  - distance = 0   → vecteurs identiques (similarité parfaite)
  - distance = 1   → vecteurs orthogonaux (aucune relation)
  - distance = 2   → vecteurs opposés
On convertit la distance en score de similarité : score = 1 - distance.
Un seuil de 0.4 (distance < 0.6) filtre les documents hors domaine.
"""

from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

SEUIL_SIMILARITE = 0.4  # En dessous de ce score, le document est ignoré
TOP_K_DEFAUT = 3


def rechercher_documents(
    requete: str,
    collection: chromadb.Collection,
    modele_embedding: SentenceTransformer,
    top_k: int = TOP_K_DEFAUT,
    seuil: float = SEUIL_SIMILARITE,
) -> list[dict[str, Any]]:
    """
    Recherche les top_k documents les plus proches de la requête.

    Retourne une liste de dicts avec les clés :
        - 'id'        : identifiant du document dans ChromaDB
        - 'document'  : texte de la question FAQ indexée
        - 'reponse'   : réponse FAQ (depuis les métadonnées)
        - 'categorie' : catégorie FAQ
        - 'score'     : score de similarité [0, 1] (1 = identique)
        - 'distance'  : distance cosinus brute retournée par ChromaDB
    """
    # Encode la requête avec le même modèle que lors de l'indexation
    vecteur_requete = modele_embedding.encode(
        [requete], normalize_embeddings=True
    ).tolist()

    resultats = collection.query(
        query_embeddings=vecteur_requete,
        n_results=min(top_k, collection.count()),  # évite l'erreur si collection petite
        include=["documents", "metadatas", "distances"],
    )

    documents_filtres = []
    ids = resultats["ids"][0]
    documents = resultats["documents"][0]
    metadatas = resultats["metadatas"][0]
    distances = resultats["distances"][0]

    for id_, doc, meta, dist in zip(ids, documents, metadatas, distances):
        score = 1.0 - dist  # conversion distance cosinus → similarité
        if score >= seuil:
            documents_filtres.append(
                {
                    "id": id_,
                    "document": doc,
                    "reponse": meta.get("reponse", ""),
                    "categorie": meta.get("categorie", ""),
                    "score": round(score, 4),
                    "distance": round(dist, 4),
                }
            )

    return documents_filtres


if __name__ == "__main__":
    from rag.base_connaissance import initialiser_base

    collection, modele = initialiser_base()
    resultats = rechercher_documents(
        "Je voudrais renvoyer un article", collection, modele
    )
    for r in resultats:
        print(f"[{r['score']:.3f}] {r['document']} → {r['reponse'][:60]}...")