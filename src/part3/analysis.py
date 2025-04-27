#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Partie 3: Exploration Sémantique et Distributionnelle
Utilisation de Word Embeddings Préentraînés avec spaCy
"""

import os
import polars as pl
import numpy as np
import itertools
import cupy as cp
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from cuml.decomposition import PCA

sns.set(style="whitegrid")

# Chargement du dataset
df = pl.read_parquet("dataset/train.parquet")
df = df.with_columns(pl.col("text").str.split(" ").alias("tokens"))

# Chargement du modèle spaCy avec vecteurs
print("Chargement du modèle spaCy en_core_web_md...")
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")
print("Modèle spaCy chargé avec succès.")

# Termes clés d'intérêt
terms = [
    "help", "alone", "pain", "hope", "friend", "feel", "not",
    "like", "want", "know", "life", "time", "people", "tired", "suicide"
]

# Construire la sélection de mots (termes clés + top-5 similaires via spaCy)
selection = set(terms)
for term in terms:
    if nlp.vocab.has_vector(term):
        vec = nlp.vocab.get_vector(term)
        keys, _, _ = nlp.vocab.vectors.most_similar(vec.reshape(1, -1), n=5)
        selection.update(nlp.vocab.strings[k] for k in keys[0])
selection = list(selection)

# Créer le répertoire pour les figures
os.makedirs("outputs/part3/plots", exist_ok=True)

# Section 1 : Exploration des similarités
for term in terms:
    if nlp.vocab.has_vector(term):
        vec = nlp.vocab.get_vector(term)
        keys, scores, _ = nlp.vocab.vectors.most_similar(vec.reshape(1, -1), n=10)
        mots = [nlp.vocab.strings[k] for k in keys[0]]
        sims = scores[0].astype(float)
        plt.figure(figsize=(8, 6))
        sns.barplot(
            x=list(sims),
            y=mots,
            hue=mots,
            palette="viridis",
            dodge=False,
            legend=False
        )
        plt.title(f"Mots les plus similaires à '{term}'")
        plt.xlabel("Score brut spaCy")
        plt.ylabel("Mot")
        plt.tight_layout()
        plt.savefig(f"outputs/part3/plots/similar_{term}.png")
        plt.close()
    else:
        print(f"Le terme '{term}' n'est pas dans le vocabulaire spaCy.")

# Section 2 : Classement global des analogies vectorielles
analogy_results = []
for a, b, c in itertools.combinations(terms, 3):
    if nlp.vocab.has_vector(a) and nlp.vocab.has_vector(b) and nlp.vocab.has_vector(c):
        vec_query = (nlp.vocab.get_vector(a) + nlp.vocab.get_vector(b) - nlp.vocab.get_vector(c))
        vec_query /= np.linalg.norm(vec_query)
        vecs = np.array([nlp.vocab.get_vector(w) for w in selection])
        vecs_norm = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        sims = vecs_norm.dot(vec_query)
        best_score = float(np.max(sims))
        analogy_results.append({"combo": f"{a}+{b}-{c}", "score": best_score})
# Trier les analogies par score décroissant
analogy_results.sort(key=lambda x: x["score"], reverse=True)
# Afficher seulement les N meilleures analogies pour améliorer la lisibilité
top_n = 10
top_analogies = analogy_results[:top_n]
plt.figure(figsize=(12, 8))
combos = [r["combo"] for r in top_analogies]
scores = [r["score"] for r in top_analogies]
ax = sns.barplot(x=scores, y=combos, hue=combos, palette="magma", dodge=False, legend=False)
plt.yticks(fontsize=8)
plt.title(f"Top {top_n} Classement des combinaisons d'analogies")
plt.xlabel("Score cosinus maximal")
plt.ylabel("Combinaison (a+b-c)")
plt.tight_layout()
plt.savefig("outputs/part3/plots/analogy_ranking.png")
plt.close()

# Section 3 : Visualisation des embeddings via PCA
vectors_gpu = cp.array([nlp.vocab.get_vector(w) for w in selection])
pca = PCA(n_components=2)
coords_gpu = pca.fit_transform(vectors_gpu)
coords = cp.asnumpy(coords_gpu)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=coords[:, 0], y=coords[:, 1], s=50)
for i, mot in enumerate(selection):
    plt.text(coords[i, 0] + 0.01, coords[i, 1] + 0.01, mot, fontsize=9)
plt.title("Projection PCA des embeddings des mots sélectionnés")
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.tight_layout()
plt.savefig("outputs/part3/plots/embeddings_pca.png")
plt.close()

print("Partie 3 terminée : tous les graphiques sont dans outputs/part3/plots.")
