"""
Exercice 0.6 — Reranking : affinage de la sélection par cross-encoder.

Le cross-encoder évalue conjointement chaque paire (requête, candidat) pour
produire un score de pertinence plus précis que le bi-encoder.

Différence clé :
  - Bi-encoder  : encode requête et document séparément → rapide, approximatif
  - Cross-encoder : encode la PAIRE (requête, document) → lent, précis

Les scores du cross-encoder sont des logits bruts (peuvent être négatifs ou
dépasser 1). On les utilise uniquement pour le classement relatif, pas comme
probabilités. Aucune normalisation nécessaire avant le tri.
"""

from typing import Any

from sentence_transformers import CrossEncoder

MODELE_CROSSENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def charger_crossencoder(nom_modele: str = MODELE_CROSSENCODER) -> CrossEncoder:
    """Charge et retourne le modèle cross-encoder."""
    return CrossEncoder(nom_modele)


def reclasser_passages(
    requete: str,
    candidats: list[dict[str, Any]],
    modele_crossencoder: CrossEncoder,
    top_k_final: int = 3,
) -> list[dict[str, Any]]:
    """
    Reclasse les candidats (résultat du bi-encoder) avec le cross-encoder.

    Args:
        requete          : texte de la requête utilisateur
        candidats        : liste de dicts avec au moins la clé 'reponse' (ou 'document')
        modele_crossencoder : instance de CrossEncoder
        top_k_final      : nombre de candidats à retourner après reranking

    Retourne les top_k_final meilleurs candidats, triés par score décroissant,
    avec un champ 'score_reranking' ajouté à chaque dict.
    """
    if not candidats:
        return []

    # Le cross-encoder attend des paires (requête, texte_du_candidat)
    # On utilise la réponse FAQ comme texte à évaluer (plus informatif que la question)
    paires = [
        (requete, c.get("reponse", c.get("document", "")))
        for c in candidats
    ]

    scores = modele_crossencoder.predict(paires)

    # Associe chaque score à son candidat
    candidats_scores = [
        {**candidat, "score_reranking": float(score)}
        for candidat, score in zip(candidats, scores)
    ]

    # Tri décroissant par score cross-encoder
    candidats_scores.sort(key=lambda x: x["score_reranking"], reverse=True)

    # Log du meilleur score (utile pour Evidently AI)
    if candidats_scores:
        meilleur = candidats_scores[0]["score_reranking"]
        print(f"[Reranking] Meilleur score cross-encoder : {meilleur:.4f}")

    return candidats_scores[:top_k_final]


if __name__ == "__main__":
    from rag.base_connaissance import initialiser_base
    from rag.recherche import rechercher_documents

    collection, modele_bi = initialiser_base()
    crossencoder = charger_crossencoder()

    requete = "Je veux retourner un article abîmé"
    candidats = rechercher_documents(requete, collection, modele_bi, top_k=10)

    print(f"\n{len(candidats)} candidats bi-encoder :")
    for c in candidats:
        print(f"  [{c['score']:.3f}] {c['document']}")

    reclasses = reclasser_passages(requete, candidats, crossencoder, top_k_final=3)

    print(f"\nTop 3 après reranking :")
    for r in reclasses:
        print(f"  [{r['score_reranking']:.4f}] {r['document']}")