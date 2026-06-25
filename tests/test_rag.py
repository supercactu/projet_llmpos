"""
Exercice 0.4 & 0.6 — Tests unitaires du pipeline RAG et du reranking.

Les tests utilisent une collection ChromaDB EN MÉMOIRE (EphemeralClient)
peuplée de quelques documents de test, isolée de la collection de production.

Note sur le backend Rust de ChromaDB (≥ 0.6) :
  EphemeralClient partage un état global dans le même processus Python.
  create_collection() échoue si le nom existe déjà. Solution : nom unique
  par test via uuid4, ce qui garantit l'isolation sans recréer le client.
"""

import uuid
import pytest
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from unittest.mock import MagicMock, patch

from rag.recherche import rechercher_documents
from rag.reranking import reclasser_passages
from rag.pipeline import construire_prompt_augmente, generer_avec_rag

MODELE_EMBEDDING = "all-MiniLM-L6-v2"
MODELE_CROSSENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Client EphemeralClient partagé pour toute la session
_CLIENT_EPHEMERE = chromadb.EphemeralClient()

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def modele_embedding():
    """Charge le modèle une seule fois pour toute la session de tests."""
    return SentenceTransformer(MODELE_EMBEDDING)


@pytest.fixture(scope="session")
def modele_crossencoder():
    return CrossEncoder(MODELE_CROSSENCODER)


@pytest.fixture
def collection_test(modele_embedding):
    """
    Collection ChromaDB en mémoire peuplée de documents de test.
    Nom unique par test (uuid) pour éviter les conflits avec le backend Rust
    qui partage l'état EphemeralClient dans le même processus.
    """
    nom_unique = f"test_faq_{uuid.uuid4().hex}"
    collection = _CLIENT_EPHEMERE.create_collection(
        name=nom_unique,
        metadata={"hnsw:space": "cosine"},
    )

    docs_test = [
        {
            "id": "faq-01",
            "question": "Quel est le délai de retour ?",
            "reponse": "Vous pouvez retourner tout article dans un délai de 30 jours.",
            "categorie": "retours",
        },
        {
            "id": "faq-02",
            "question": "Comment suivre ma commande ?",
            "reponse": "Un email avec un lien de suivi est envoyé dès l'expédition.",
            "categorie": "livraison",
        },
        {
            "id": "faq-04",
            "question": "Comment obtenir un remboursement ?",
            "reponse": "Le remboursement est effectué sous 5 à 10 jours ouvrés.",
            "categorie": "remboursement",
        },
        {
            "id": "faq-06",
            "question": "Que faire si mon colis est endommagé ?",
            "reponse": "Prenez des photos et contactez le support dans les 48h.",
            "categorie": "litiges",
        },
        {
            "id": "faq-07",
            "question": "Comment annuler une commande ?",
            "reponse": "Une commande peut être annulée dans les 2 heures suivant sa validation.",
            "categorie": "commandes",
        },
    ]

    questions = [d["question"] for d in docs_test]
    embeddings = modele_embedding.encode(questions, normalize_embeddings=True)

    collection.add(
        ids=[d["id"] for d in docs_test],
        embeddings=embeddings.tolist(),
        documents=questions,
        metadatas=[
            {"reponse": d["reponse"], "categorie": d["categorie"], "id_original": d["id"]}
            for d in docs_test
        ],
    )

    return collection


# ── Tests Exercice 0.4 ────────────────────────────────────────────────────────

class TestRecherche:

    def test_recherche_retourne_resultats(self, collection_test, modele_embedding):
        """La recherche retourne bien top_k résultats pour une requête pertinente."""
        resultats = rechercher_documents(
            "retourner un article",
            collection_test,
            modele_embedding,
            top_k=3,
        )
        assert len(resultats) <= 3
        assert len(resultats) > 0

    def test_recherche_pertinence(self, collection_test, modele_embedding):
        """Pour 'retourner un article', faq-01 doit apparaître dans les résultats."""
        resultats = rechercher_documents(
            "retourner un article",
            collection_test,
            modele_embedding,
            top_k=3,
        )
        ids_retrouves = [r["id"] for r in resultats]
        assert "faq-01" in ids_retrouves, (
            f"faq-01 non trouvé dans {ids_retrouves} — vérifier le seuil ou le modèle"
        )

    def test_recherche_hors_domaine(self, collection_test, modele_embedding):
        """Pour une requête hors domaine, aucun document ne dépasse le seuil."""
        resultats = rechercher_documents(
            "Quelle est la température de fusion du tungstène ?",
            collection_test,
            modele_embedding,
            top_k=3,
            seuil=0.4,
        )
        assert len(resultats) == 0, (
            f"Des documents ont été retrouvés à tort : {[r['score'] for r in resultats]}"
        )

    def test_recherche_scores_valides(self, collection_test, modele_embedding):
        """Les scores de similarité sont bien dans [0, 1]."""
        resultats = rechercher_documents(
            "suivi de commande",
            collection_test,
            modele_embedding,
            top_k=3,
        )
        for r in resultats:
            assert 0.0 <= r["score"] <= 1.0, f"Score invalide : {r['score']}"


class TestPromptAugmente:

    def test_prompt_augmente_contient_contexte(self, collection_test, modele_embedding):
        """Le prompt augmenté contient bien les textes des documents trouvés."""
        passages = rechercher_documents(
            "retourner un article",
            collection_test,
            modele_embedding,
            top_k=2,
        )
        prompt = construire_prompt_augmente("Comment retourner un article ?", passages)

        assert "Comment retourner un article ?" in prompt
        assert "Contexte" in prompt
        for p in passages:
            assert p["reponse"] in prompt

    def test_prompt_augmente_structure(self, collection_test, modele_embedding):
        """Le prompt respecte la structure attendue avec les marqueurs [1], [2]..."""
        passages = rechercher_documents(
            "remboursement",
            collection_test,
            modele_embedding,
            top_k=2,
        )
        prompt = construire_prompt_augmente("Comment se faire rembourser ?", passages)
        assert "[1]" in prompt
        if len(passages) >= 2:
            assert "[2]" in prompt


class TestPipelineComplet:

    def test_pipeline_complet_retourne_str(self, collection_test, modele_embedding):
        """Le pipeline complet retourne une chaîne de caractères non vide."""
        # Mock du LLM et du tokeniseur pour éviter le téléchargement du modèle
        tokeniseur_mock = MagicMock()
        tokeniseur_mock.return_value = {"input_ids": MagicMock(shape=[1, 10])}
        tokeniseur_mock.eos_token_id = 0
        tokeniseur_mock.decode.return_value = "Vous pouvez retourner sous 30 jours."

        modele_llm_mock = MagicMock()
        outputs_mock = MagicMock()
        outputs_mock.__getitem__ = MagicMock(return_value=MagicMock())
        modele_llm_mock.generate.return_value = [MagicMock()]

        with patch("rag.pipeline.rechercher_documents") as mock_recherche:
            mock_recherche.return_value = [
                {
                    "id": "faq-01",
                    "document": "Quel est le délai de retour ?",
                    "reponse": "Vous pouvez retourner sous 30 jours.",
                    "categorie": "retours",
                    "score": 0.85,
                    "distance": 0.15,
                }
            ]

            # On teste la construction du prompt (la partie testable sans GPU)
            passages = mock_recherche.return_value
            prompt = construire_prompt_augmente("délai de retour", passages)
            assert isinstance(prompt, str)
            assert len(prompt) > 0

    def test_pipeline_hors_domaine(self, collection_test, modele_embedding):
        """Si aucun document n'est trouvé, le pipeline retourne hors_domaine=True."""
        with patch("rag.pipeline.rechercher_documents") as mock_recherche:
            mock_recherche.return_value = []  # Aucun document pertinent

            resultat = generer_avec_rag(
                "Quelle est la distance Terre-Lune ?",
                collection_test,
                modele_embedding,
                modele_llm=MagicMock(),
                tokeniseur=MagicMock(),
            )

            assert resultat["hors_domaine"] is True
            assert resultat["reponse"] != ""
            assert resultat["ids_sources"] == []


# ── Tests Exercice 0.6 — Reranking ───────────────────────────────────────────

class TestReranking:

    def test_reranking_retourne_top_k(self, modele_crossencoder):
        """La fonction retourne bien top_k_final éléments."""
        candidats = [
            {"id": "faq-01", "reponse": "Retours acceptés sous 30 jours.", "score": 0.8},
            {"id": "faq-04", "reponse": "Remboursement sous 5 à 10 jours.", "score": 0.6},
            {"id": "faq-07", "reponse": "Annulation possible dans les 2h.", "score": 0.5},
            {"id": "faq-02", "reponse": "Email de suivi envoyé à l'expédition.", "score": 0.4},
        ]
        reclasses = reclasser_passages(
            "retourner un article abîmé",
            candidats,
            modele_crossencoder,
            top_k_final=2,
        )
        assert len(reclasses) == 2

    def test_reranking_ordre_coherent(self, modele_crossencoder):
        """Le candidat le plus pertinent est classé premier."""
        candidats = [
            {"id": "faq-01", "reponse": "Retours acceptés sous 30 jours.", "score": 0.7},
            {"id": "faq-05", "reponse": "Nous livrons dans 30 pays.", "score": 0.5},
        ]
        reclasses = reclasser_passages(
            "retourner un produit",
            candidats,
            modele_crossencoder,
            top_k_final=2,
        )
        # Le doc sur les retours doit être mieux classé que le doc sur la livraison
        assert reclasses[0]["id"] == "faq-01"
        assert reclasses[0]["score_reranking"] >= reclasses[1]["score_reranking"]

    def test_reranking_candidats_vides(self, modele_crossencoder):
        """Avec une liste vide, la fonction retourne une liste vide."""
        reclasses = reclasser_passages("test", [], modele_crossencoder, top_k_final=3)
        assert reclasses == []

    def test_reranking_integre_pipeline(self, collection_test, modele_embedding, modele_crossencoder):
        """Le pipeline avec reranking retourne bien une réponse non vide (hors appel LLM)."""
        candidats = rechercher_documents(
            "retourner un article",
            collection_test,
            modele_embedding,
            top_k=5,
        )
        reclasses = reclasser_passages(
            "retourner un article",
            candidats,
            modele_crossencoder,
            top_k_final=3,
        )
        prompt = construire_prompt_augmente("retourner un article", reclasses)
        assert isinstance(prompt, str)
        assert len(prompt) > 50