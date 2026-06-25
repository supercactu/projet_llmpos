"""
Exercice 0.3 & 0.6 — Pipeline RAG complet avec reranking.

Flux :
  1. Recherche bi-encoder → top_k_biencoder candidats (ChromaDB)
  2. Reranking cross-encoder → top_k_final meilleurs passages
  3. Construction du prompt augmenté avec le contexte injecté
  4. Génération LLM à partir du prompt augmenté

Cas dégénéré : si aucun document ne dépasse le seuil de pertinence,
le pipeline répond "Je n'ai pas l'information." sans appeler le LLM,
ce qui évite les hallucinations hors domaine.
"""

from typing import Any

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

from rag.recherche import rechercher_documents
from rag.reranking import reclasser_passages

# Paramètres du pipeline
TOP_K_BIENCODER = 10   # Candidats récupérés par le bi-encoder
TOP_K_FINAL = 3         # Passages retenus après reranking
SEUIL_PERTINENCE = 0.4  # Score minimum pour considérer un document pertinent

REPONSE_HORS_DOMAINE = (
    "Je n'ai pas l'information pour répondre à cette question "
    "avec les données disponibles."
)


def construire_prompt_augmente(
    prompt_utilisateur: str,
    passages: list[dict[str, Any]],
) -> str:
    """
    Construit le prompt augmenté en injectant le contexte des passages retrouvés.

    Structure :
        [Instruction système]
        Contexte :
        [1] texte du passage 1
        [2] texte du passage 2
        Question : <prompt utilisateur>
        Réponse :
    """
    contexte_lignes = []
    for i, passage in enumerate(passages, start=1):
        # On injecte la réponse FAQ (plus informative que la question indexée)
        texte = passage.get("reponse", passage.get("document", ""))
        contexte_lignes.append(f"[{i}] {texte}")

    contexte = "\n".join(contexte_lignes)

    prompt = (
        "Tu es un assistant service client. "
        "Réponds uniquement en te basant sur le contexte fourni. "
        "Si la réponse n'est pas dans le contexte, réponds : "
        "\"Je n'ai pas l'information.\"\n\n"
        f"Contexte :\n{contexte}\n\n"
        f"Question : {prompt_utilisateur}\n"
        "Réponse :"
    )
    return prompt


def generer_avec_rag(
    prompt_utilisateur: str,
    collection: chromadb.Collection,
    modele_embedding: SentenceTransformer,
    modele_llm: Any,
    tokeniseur: Any,
    modele_crossencoder: CrossEncoder | None = None,
    nb_tokens_max: int = 200,
) -> dict[str, Any]:
    """
    Pipeline RAG complet : recherche → (reranking) → génération.

    Retourne un dict avec :
        - 'reponse'         : texte généré par le LLM
        - 'prompt_augmente' : prompt complet envoyé au LLM
        - 'passages_utilises': liste des passages injectés
        - 'ids_sources'     : IDs des documents sources
        - 'hors_domaine'    : True si aucun document pertinent trouvé
    """
    # ── Étape 1 : Recherche bi-encoder ──────────────────────────────────────
    top_k_bi = TOP_K_BIENCODER if modele_crossencoder else TOP_K_FINAL
    candidats = rechercher_documents(
        requete=prompt_utilisateur,
        collection=collection,
        modele_embedding=modele_embedding,
        top_k=top_k_bi,
        seuil=SEUIL_PERTINENCE,
    )

    # Cas dégénéré : aucun document pertinent
    if not candidats:
        return {
            "reponse": REPONSE_HORS_DOMAINE,
            "prompt_augmente": None,
            "passages_utilises": [],
            "ids_sources": [],
            "hors_domaine": True,
        }

    # ── Étape 2 : Reranking (optionnel) ────────────────────────────────────
    if modele_crossencoder is not None:
        passages_finaux = reclasser_passages(
            requete=prompt_utilisateur,
            candidats=candidats,
            modele_crossencoder=modele_crossencoder,
            top_k_final=TOP_K_FINAL,
        )
    else:
        passages_finaux = candidats[:TOP_K_FINAL]

    # ── Étape 3 : Construction du prompt augmenté ───────────────────────────
    prompt_augmente = construire_prompt_augmente(prompt_utilisateur, passages_finaux)

    # ── Étape 4 : Génération LLM ────────────────────────────────────────────
    inputs = tokeniseur(prompt_augmente, return_tensors="pt")
    outputs = modele_llm.generate(
        **inputs,
        max_new_tokens=nb_tokens_max,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokeniseur.eos_token_id,
    )
    # On ne garde que les tokens nouvellement générés (pas le prompt)
    tokens_generes = outputs[0][inputs["input_ids"].shape[1]:]
    reponse = tokeniseur.decode(tokens_generes, skip_special_tokens=True).strip()

    return {
        "reponse": reponse,
        "prompt_augmente": prompt_augmente,
        "passages_utilises": passages_finaux,
        "ids_sources": [p["id"] for p in passages_finaux],
        "hors_domaine": False,
    }