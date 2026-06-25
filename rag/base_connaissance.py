"""
Exercice 0.1 — Création et alimentation de la base de connaissance ChromaDB.

Stratégie d'indexation : on encode la QUESTION (et non la réponse) de chaque
entrée FAQ. Ainsi, lors de la recherche, le vecteur de la requête utilisateur
(ex. "je veux renvoyer un article") est comparé aux vecteurs des questions FAQ
("Quel est le délai de retour ?"), dont le sens est plus proche que celui des
réponses — qui contiennent des détails factuels sans forcément reprendre les
mots de l'utilisateur.
"""

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

MODELE_EMBEDDING = "all-MiniLM-L6-v2"
NOM_COLLECTION = "faq_service_client"
CHEMIN_CHROMA = "data/chroma_db"


def charger_faq(chemin_jsonl: str) -> list[dict]:
    """Charge les entrées FAQ depuis un fichier JSONL."""
    entrees = []
    with open(chemin_jsonl, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if ligne:
                entrees.append(json.loads(ligne))
    return entrees


def creer_collection(chemin_chroma: str = CHEMIN_CHROMA) -> chromadb.Collection:
    """Initialise un client ChromaDB persistant et retourne la collection FAQ."""
    client = chromadb.PersistentClient(path=chemin_chroma)
    # cosine : adapté aux embeddings normalisés (all-MiniLM produit des vecteurs
    # L2-normalisés). La distance cosinus vaut 0 pour des vecteurs identiques
    # et 2 pour des vecteurs opposés.
    collection = client.get_or_create_collection(
        name=NOM_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def alimenter_collection(
    collection: chromadb.Collection,
    entrees_faq: list[dict],
    modele: SentenceTransformer,
) -> None:
    """
    Calcule les embeddings et insère les documents dans la collection.
    Vérifie d'abord si la collection est déjà peuplée pour éviter les doublons.
    """
    if collection.count() >= len(entrees_faq):
        print(f"Collection déjà peuplée ({collection.count()} documents). Skipping.")
        return

    # On encode les QUESTIONS pour que la similarité soit calculée entre
    # la requête utilisateur et la question FAQ (même registre sémantique).
    questions = [e["question"] for e in entrees_faq]
    embeddings = modele.encode(questions, normalize_embeddings=True)

    collection.add(
        ids=[e["id"] for e in entrees_faq],
        embeddings=embeddings.tolist(),
        documents=[e["question"] for e in entrees_faq],
        metadatas=[
            {
                "reponse": e["reponse"],
                "categorie": e["categorie"],
                "id_original": e["id"],
            }
            for e in entrees_faq
        ],
    )
    print(f"{len(entrees_faq)} documents indexés dans '{NOM_COLLECTION}'.")


def initialiser_base(
    chemin_jsonl: str = "data/faq_service_client.jsonl",
    chemin_chroma: str = CHEMIN_CHROMA,
    nom_modele: str = MODELE_EMBEDDING,
) -> tuple[chromadb.Collection, SentenceTransformer]:
    """
    Point d'entrée principal : charge la FAQ, initialise ChromaDB et le modèle
    d'embedding, puis alimente la collection si nécessaire.

    Retourne (collection, modele_embedding) pour être réutilisés dans le pipeline.
    """
    modele = SentenceTransformer(nom_modele)
    collection = creer_collection(chemin_chroma)
    entrees = charger_faq(chemin_jsonl)
    alimenter_collection(collection, entrees, modele)
    return collection, modele


if __name__ == "__main__":
    col, mod = initialiser_base()
    print(f"Base prête — {col.count()} documents dans la collection.")