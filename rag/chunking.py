"""
Exercice 0.5 — Chunking : découpage des documents longs.

Le chunking divise un document en passages (chunks) de taille fixe avec un
chevauchement (overlap) pour ne pas couper les phrases à la frontière entre
deux chunks et préserver le contexte de transition.

Fenêtre glissante :
  - Début du chunk i : i * (taille_chunk - chevauchement)
  - Fin du chunk i   : début + taille_chunk
  - Incrément entre deux débuts consécutifs = taille_chunk - chevauchement
"""

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


def decouper_document(
    texte: str,
    taille_chunk: int = 200,
    chevauchement: int = 40,
) -> list[str]:
    """
    Découpe `texte` en chunks de `taille_chunk` mots avec `chevauchement` mots
    partagés entre deux chunks consécutifs.

    Cas limites :
    - Texte vide          → []
    - Texte < taille_chunk → [texte entier]
    - chevauchement >= taille_chunk → ValueError (boucle infinie)
    """
    if not texte or not texte.strip():
        return []

    if chevauchement >= taille_chunk:
        raise ValueError(
            f"chevauchement ({chevauchement}) doit être < taille_chunk ({taille_chunk})"
        )

    mots = texte.split()

    if len(mots) <= taille_chunk:
        return [texte]

    chunks = []
    increment = taille_chunk - chevauchement
    debut = 0

    while debut < len(mots):
        fin = debut + taille_chunk
        chunk_mots = mots[debut:fin]
        chunks.append(" ".join(chunk_mots))
        debut += increment

    return chunks


def indexer_documents_longs(
    chemin_jsonl: str,
    collection: chromadb.Collection,
    modele_embedding: SentenceTransformer,
    taille_chunk: int = 200,
    chevauchement: int = 40,
) -> int:
    """
    Charge un fichier JSONL de documents longs, découpe chaque document en
    chunks et les insère dans ChromaDB avec l'ID du document parent en
    métadonnée.

    Format attendu du JSONL : {"id": "doc-01", "texte": "...", "titre": "..."}

    Retourne le nombre total de chunks indexés.
    """
    chemin = Path(chemin_jsonl)
    if not chemin.exists():
        raise FileNotFoundError(f"Fichier introuvable : {chemin_jsonl}")

    nb_chunks_total = 0

    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne:
                continue

            doc = json.loads(ligne)
            id_doc = doc["id"]
            texte = doc.get("texte", doc.get("content", ""))

            chunks = decouper_document(texte, taille_chunk, chevauchement)

            if not chunks:
                continue

            # IDs uniques et traçables : "{id_document}__chunk_{numéro}"
            ids_chunks = [f"{id_doc}__chunk_{i}" for i in range(len(chunks))]

            # Vérifie les doublons : skip les chunks déjà présents
            existants = set(collection.get(ids=ids_chunks)["ids"])
            nouveaux_idx = [
                i for i, cid in enumerate(ids_chunks) if cid not in existants
            ]

            if not nouveaux_idx:
                continue

            chunks_nouveaux = [chunks[i] for i in nouveaux_idx]
            ids_nouveaux = [ids_chunks[i] for i in nouveaux_idx]
            embeddings = modele_embedding.encode(
                chunks_nouveaux, normalize_embeddings=True
            )

            metadatas = [
                {
                    "id_document_parent": id_doc,
                    "numero_chunk": nouveaux_idx[j],
                    "titre": doc.get("titre", ""),
                }
                for j in range(len(nouveaux_idx))
            ]

            collection.add(
                ids=ids_nouveaux,
                embeddings=embeddings.tolist(),
                documents=chunks_nouveaux,
                metadatas=metadatas,
            )

            nb_chunks_total += len(nouveaux_idx)

    print(f"{nb_chunks_total} chunks indexés depuis '{chemin_jsonl}'.")
    return nb_chunks_total


if __name__ == "__main__":
    # Exemple d'utilisation rapide
    texte_test = " ".join([f"mot{i}" for i in range(100)])
    chunks = decouper_document(texte_test, taille_chunk=30, chevauchement=10)
    print(f"{len(chunks)} chunks générés :")
    for i, c in enumerate(chunks):
        print(f"  [{i}] {c[:50]}...")