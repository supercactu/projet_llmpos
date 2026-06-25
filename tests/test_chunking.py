"""
Exercice 0.5 — Tests unitaires du module de chunking.
"""

import uuid
import pytest
import chromadb
from sentence_transformers import SentenceTransformer

from rag.chunking import decouper_document, indexer_documents_longs

MODELE_EMBEDDING = "all-MiniLM-L6-v2"

_CLIENT_EPHEMERE = chromadb.EphemeralClient()


@pytest.fixture(scope="session")
def modele_embedding():
    return SentenceTransformer(MODELE_EMBEDDING)


@pytest.fixture
def collection_memoire():
    """
    Collection ChromaDB en mémoire avec nom unique par test.
    Nécessaire car le backend Rust partage l'état EphemeralClient
    dans le même processus — create_collection() échoue si le nom existe déjà.
    """
    nom_unique = f"test_chunks_{uuid.uuid4().hex}"
    return _CLIENT_EPHEMERE.create_collection(
        name=nom_unique,
        metadata={"hnsw:space": "cosine"},
    )


# ── Tests de decouper_document ────────────────────────────────────────────────

class TestDecouperDocument:

    def test_decoupe_sans_chevauchement(self):
        """100 mots, chunks de 30, sans chevauchement → 4 chunks (30+30+30+10)."""
        texte = " ".join([f"mot{i}" for i in range(100)])
        chunks = decouper_document(texte, taille_chunk=30, chevauchement=0)
        assert len(chunks) == 4
        # Vérifie que tous les mots sont bien présents
        mots_reconstruits = " ".join(chunks).split()
        assert len(mots_reconstruits) == 100

    def test_decoupe_avec_chevauchement(self):
        """Les mots de transition apparaissent bien dans deux chunks consécutifs."""
        texte = " ".join([f"mot{i}" for i in range(50)])
        chunks = decouper_document(texte, taille_chunk=20, chevauchement=5)

        # Les 5 derniers mots du chunk 0 doivent être les 5 premiers du chunk 1
        fin_chunk0 = chunks[0].split()[-5:]
        debut_chunk1 = chunks[1].split()[:5]
        assert fin_chunk0 == debut_chunk1

    def test_chunk_vide(self):
        """Un texte vide retourne une liste vide."""
        assert decouper_document("") == []
        assert decouper_document("   ") == []

    def test_texte_plus_court_que_chunk(self):
        """Un texte de 50 mots avec taille_chunk=200 retourne un seul chunk."""
        texte = " ".join([f"mot{i}" for i in range(50)])
        chunks = decouper_document(texte, taille_chunk=200, chevauchement=20)
        assert len(chunks) == 1
        assert chunks[0] == texte

    def test_chevauchement_invalide(self):
        """Un chevauchement >= taille_chunk doit lever ValueError."""
        with pytest.raises(ValueError):
            decouper_document("un deux trois", taille_chunk=10, chevauchement=10)
        with pytest.raises(ValueError):
            decouper_document("un deux trois", taille_chunk=10, chevauchement=15)

    def test_decoupe_texte_exact(self):
        """Un texte exactement de taille taille_chunk → 1 seul chunk."""
        texte = " ".join([f"mot{i}" for i in range(30)])
        chunks = decouper_document(texte, taille_chunk=30, chevauchement=5)
        assert len(chunks) == 1

    def test_chunks_non_vides(self):
        """Tous les chunks générés sont non vides."""
        texte = " ".join([f"mot{i}" for i in range(80)])
        chunks = decouper_document(texte, taille_chunk=25, chevauchement=5)
        for chunk in chunks:
            assert chunk.strip() != ""


# ── Tests d'indexer_documents_longs ──────────────────────────────────────────

class TestIndexerDocumentsLongs:

    def test_metadonnees_source(self, collection_memoire, modele_embedding, tmp_path):
        """Chaque chunk indexé dans ChromaDB a bien l'ID du document parent."""
        jsonl_content = (
            '{"id": "doc-test-01", "texte": "' +
            " ".join([f"mot{i}" for i in range(60)]) +
            '", "titre": "Doc test"}\n'
        )
        fichier = tmp_path / "docs_test.jsonl"
        fichier.write_text(jsonl_content, encoding="utf-8")

        nb = indexer_documents_longs(
            str(fichier),
            collection_memoire,
            modele_embedding,
            taille_chunk=30,
            chevauchement=5,
        )

        assert nb > 0

        # Vérifie les métadonnées de tous les chunks indexés
        resultats = collection_memoire.get(
            where={"id_document_parent": "doc-test-01"},
            include=["metadatas"],
        )
        assert len(resultats["ids"]) > 0
        for meta in resultats["metadatas"]:
            assert meta["id_document_parent"] == "doc-test-01"

    def test_ids_chunks_uniques_et_tracables(self, modele_embedding, tmp_path):
        """Les IDs des chunks suivent la convention '{id_doc}__chunk_{n}'."""
        nom_unique = f"test_ids_{uuid.uuid4().hex}"
        collection = _CLIENT_EPHEMERE.create_collection(
            name=nom_unique,
            metadata={"hnsw:space": "cosine"},
        )

        jsonl_content = (
            '{"id": "doc-abc", "texte": "' +
            " ".join([f"w{i}" for i in range(70)]) +
            '", "titre": "Test"}\n'
        )
        fichier = tmp_path / "test_ids.jsonl"
        fichier.write_text(jsonl_content, encoding="utf-8")

        indexer_documents_longs(
            str(fichier), collection, modele_embedding,
            taille_chunk=30, chevauchement=5,
        )

        ids = collection.get()["ids"]
        for id_ in ids:
            assert id_.startswith("doc-abc__chunk_"), f"ID inattendu : {id_}"

    def test_fichier_introuvable(self, collection_memoire, modele_embedding):
        """Un fichier inexistant lève FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            indexer_documents_longs(
                "chemin/inexistant.jsonl",
                collection_memoire,
                modele_embedding,
            )