# -*- coding: utf-8 -*-
"""
Script pour la Partie 1 de l'exercice :
Chargement, Exploration Fréquentielle et Traitement Linguistique Complémentaire
sur le dataset suicide_prediction_dataset_phr.
"""

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet # Pour obtenir les POS tags pour le lemmatiseur
from nltk.probability import FreqDist
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.collocations import TrigramAssocMeasures, TrigramCollocationFinder
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import numpy as np # Pour le log dans la loi de Zipf

# Configuration initiale
# Mise à jour pour utiliser le fichier Parquet
DATASET_PATH = 'dataset/train.parquet'
OUTPUT_DIR = 'outputs/part1/plots'

# Téléchargement des ressources NLTK nécessaires
print("Téléchargement des ressources NLTK (si nécessaire)...")
try:
    nltk.data.find('tokenizers/punkt')
except LookupError: # Utiliser LookupError qui est levée par find()
    print("Ressource 'punkt' non trouvée. Téléchargement...")
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Ressource 'wordnet' non trouvée. Téléchargement...")
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    print("Ressource 'omw-1.4' non trouvée. Téléchargement...")
    nltk.download('omw-1.4')
try:
    nltk.download('averaged_perceptron_tagger_eng')
except LookupError:
    print("Ressource 'averaged_perceptron_tagger' non trouvée. Téléchargement...")
    nltk.download('averaged_perceptron_tagger_eng')
# Vérifier également punkt_tab qui peut être nécessaire pour la tokenisation de phrases
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Ressource 'punkt_tab' non trouvée. Téléchargement...")
    nltk.download('punkt_tab')
print("Ressources NLTK vérifiées/téléchargées.")


# Créer le répertoire de sortie s'il n'existe pas
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("--- Partie 1 : Chargement, Exploration et Traitement ---")

# 1. Chargement et Exploration Initiale
print("\n[1. Chargement et Exploration Initiale]")
try:
    # Charger le dataset avec pandas depuis le fichier Parquet
    df = pd.read_parquet(DATASET_PATH)
    print(f"Dataset chargé depuis : {DATASET_PATH}")
    print("Aperçu des premières lignes :")
    print(df.head())
    print("\nInformations sur le DataFrame :")
    df.info()

    # Vérifier les colonnes attendues
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Les colonnes 'text' et 'label' attendues ne sont pas présentes.")
    print("\nColonnes présentes : OK ('text', 'label')")

    # Analyser la distribution des étiquettes (supposant 'suicide' et 'non-suicide')
    print("\nDistribution des étiquettes ('label') :")
    label_distribution = df['label'].value_counts(normalize=True) * 100
    print(label_distribution)
    print(f"\nNombre total d'échantillons : {len(df)}")

    # Visualisation de la distribution des étiquettes
    plt.figure(figsize=(6, 4))
    # Utiliser les labels textuels pour x et hue
    sns.countplot(x='label', data=df, hue='label', palette='viridis', legend=False, order=['non-suicide', 'suicide']) # Ordonner pour la clarté
    plt.title('Distribution des Étiquettes (Suicide vs Non-Suicide)')
    plt.xlabel('Étiquette')
    plt.ylabel('Nombre d\'occurrences')
    plot_path = os.path.join(OUTPUT_DIR, '1_label_distribution.png')
    plt.savefig(plot_path)
    print(f"Graphique de distribution sauvegardé : {plot_path}")
    # plt.show() # Décommenter pour afficher interactivement si nécessaire

except FileNotFoundError:
    print(f"ERREUR : Le fichier dataset '{DATASET_PATH}' n'a pas été trouvé.")
    exit()
except Exception as e:
    print(f"Une erreur est survenue lors du chargement ou de l'exploration initiale : {e}")
    exit()

# 2. Analyse Fréquentielle sur Données Pré-nettoyées
print("\n[2. Analyse Fréquentielle sur Données Pré-nettoyées]")

# Tokenisation de tous les textes
# S'assurer que les textes sont bien des chaînes de caractères et gérer les NaN potentiels
df['text'] = df['text'].astype(str)
all_texts = ' '.join(df['text'])
print("Tokenisation en cours...")
tokens = word_tokenize(all_texts)
print(f"Nombre total de tokens (bruts) : {len(tokens)}")

# Calcul de la fréquence des tokens
fdist = FreqDist(tokens)
print("\nFréquence des tokens calculée.")
top_n = 30
print(f"\n{top_n} tokens les plus fréquents (après pré-nettoyage initial) :")
print(fdist.most_common(top_n))

# Visualisation des fréquences (Barplot)
plt.figure(figsize=(12, 8))
fdist.plot(top_n, cumulative=False, title=f'Top {top_n} Mots les Plus Fréquents (Pré-nettoyage)')
plot_path_bar = os.path.join(OUTPUT_DIR, '2a_freq_dist_bar.png')
plt.savefig(plot_path_bar)
print(f"Graphique barplot des fréquences sauvegardé : {plot_path_bar}")
# plt.show()

# Visualisation des fréquences (Nuage de mots)
print("\nGénération du nuage de mots...")
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(fdist)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Nuage de Mots (Pré-nettoyage)')
plot_path_wc = os.path.join(OUTPUT_DIR, '2b_freq_dist_wordcloud.png')
plt.savefig(plot_path_wc)
print(f"Nuage de mots sauvegardé : {plot_path_wc}")
# plt.show()

# Commentaire sur les termes dominants :
# (À ajouter manuellement après exécution et analyse des graphiques)
# Les termes les plus fréquents après le nettoyage initial donnent une première idée
# des sujets abordés. Il sera intéressant de comparer avec les fréquences
# après lemmatisation/stemming. On remarque déjà la présence de termes comme 'not',
# qui a été conservé intentionnellement.

# 3. Traitement Linguistique Complémentaire : Lemmatisation
print("\n[3. Traitement Linguistique Complémentaire : Lemmatisation]")

# Choix : Lemmatisation avec NLTK WordNetLemmatizer.
# Justification : La lemmatisation réduit les mots à leur forme de base (lemme)
# tout en conservant le sens, ce qui est préférable au stemming (qui coupe
# simplement les terminaisons et peut créer des mots inexistants).
# Cela aide à regrouper différentes formes d'un même mot (ex: 'running', 'ran' -> 'run').
# NLTK est déjà utilisé et fournit un lemmatiseur basé sur WordNet.
# Pour une meilleure lemmatisation, on utilise les POS tags.

lemmatizer = WordNetLemmatizer()

# Fonction pour obtenir le POS tag compatible avec WordNetLemmatizer
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN) # Default to NOUN

# Fonction pour lemmatiser un texte (tokenisé)
def lemmatize_text(text):
    tokens = word_tokenize(text)
    # Lemmatiser chaque token en utilisant son POS tag
    lemmatized_tokens = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokens]
    return lemmatized_tokens

print("Application de la lemmatisation sur la colonne 'text'...")
# Appliquer la fonction sur chaque texte du DataFrame
# Cela peut prendre un certain temps sur un grand dataset
df['lemmatized_tokens'] = df['text'].apply(lemmatize_text)
print("Lemmatisation terminée.")
print("Aperçu des tokens lemmatisés pour les premières lignes :")
print(df[['text', 'lemmatized_tokens']].head())

# 4. Analyse Fréquentielle Post-Lemmatisation
print("\n[4. Analyse Fréquentielle Post-Lemmatisation]")

# Regrouper tous les tokens lemmatisés
all_lemmatized_tokens = [token for sublist in df['lemmatized_tokens'] for token in sublist]
print(f"Nombre total de tokens lemmatisés : {len(all_lemmatized_tokens)}")

# Calculer la nouvelle distribution de fréquence
fdist_lemmatized = FreqDist(all_lemmatized_tokens)
print("\nFréquence des tokens lemmatisés calculée.")
print(f"\n{top_n} tokens lemmatisés les plus fréquents :")
print(fdist_lemmatized.most_common(top_n))

# Visualisation des fréquences post-lemmatisation (Barplot)
plt.figure(figsize=(12, 8))
fdist_lemmatized.plot(top_n, cumulative=False, title=f'Top {top_n} Mots les Plus Fréquents (Après Lemmatisation)')
plot_path_bar_lemma = os.path.join(OUTPUT_DIR, '4_freq_dist_bar_lemmatized.png')
plt.savefig(plot_path_bar_lemma)
print(f"Graphique barplot des fréquences lemmatisées sauvegardé : {plot_path_bar_lemma}")
# plt.show()

# Commentaire sur la comparaison :
# (À ajouter manuellement après exécution et analyse des graphiques)
# La lemmatisation a regroupé différentes formes de mots (ex: 'feel', 'feeling').
# Il faut comparer ce graphique avec le précédent (2a) pour voir si le classement
# des mots a changé de manière significative et si de nouveaux termes émergent
# comme étant plus importants une fois normalisés. Par exemple, le lemme 'be'
# (regroupant 'is', 'am', 'are', 'was', 'were'...) pourrait monter dans le classement.

# 5. Analyse des N-grammes (sur tokens lemmatisés)
print("\n[5. Analyse des N-grammes]")

# Utiliser les tokens lemmatisés déjà calculés (all_lemmatized_tokens)

# Bi-grammes
print("Calcul des bi-grammes fréquents...")
bigram_measures = BigramAssocMeasures()
finder_bi = BigramCollocationFinder.from_words(all_lemmatized_tokens)
# Filtrer les n-grammes qui apparaissent moins de 3 fois (optionnel, pour la pertinence)
finder_bi.apply_freq_filter(3)
# Obtenir les N bi-grammes les plus fréquents
top_n_bigrams = 20
frequent_bigrams = finder_bi.nbest(bigram_measures.pmi, top_n_bigrams) # Ou utiliser raw_freq
print(f"\n{top_n_bigrams} bi-grammes les plus fréquents (basé sur PMI ou fréquence) :")
print(frequent_bigrams)

# Préparer les données pour la visualisation Seaborn (fréquence brute)
bigram_freq = finder_bi.ngram_fd.most_common(top_n_bigrams)
bigram_labels = [' '.join(gram) for gram, freq in bigram_freq]
bigram_counts = [freq for gram, freq in bigram_freq]

# Visualisation des bi-grammes
plt.figure(figsize=(12, 8))
# Modifier l'appel pour suivre les recommandations de Seaborn v0.14+
sns.barplot(x=bigram_counts, y=bigram_labels, hue=bigram_labels, palette='mako', legend=False)
plt.title(f'Top {top_n_bigrams} Bi-grammes les Plus Fréquents (Après Lemmatisation)')
plt.xlabel('Fréquence')
plt.ylabel('Bi-gramme')
plot_path_bigram = os.path.join(OUTPUT_DIR, '5a_bigram_freq.png')
plt.tight_layout() # Ajuster pour éviter que les labels se chevauchent
plt.savefig(plot_path_bigram)
print(f"Graphique des bi-grammes sauvegardé : {plot_path_bigram}")
# plt.show()

# Tri-grammes
print("\nCalcul des tri-grammes fréquents...")
trigram_measures = TrigramAssocMeasures()
finder_tri = TrigramCollocationFinder.from_words(all_lemmatized_tokens)
finder_tri.apply_freq_filter(3) # Filtrer
top_n_trigrams = 20
frequent_trigrams = finder_tri.nbest(trigram_measures.pmi, top_n_trigrams) # Ou utiliser raw_freq
print(f"\n{top_n_trigrams} tri-grammes les plus fréquents (basé sur PMI ou fréquence) :")
print(frequent_trigrams)

# Préparer les données pour la visualisation Seaborn (fréquence brute)
trigram_freq = finder_tri.ngram_fd.most_common(top_n_trigrams)
trigram_labels = [' '.join(gram) for gram, freq in trigram_freq]
trigram_counts = [freq for gram, freq in trigram_freq]

# Visualisation des tri-grammes
plt.figure(figsize=(12, 8))
# Modifier l'appel pour suivre les recommandations de Seaborn v0.14+
sns.barplot(x=trigram_counts, y=trigram_labels, hue=trigram_labels, palette='rocket', legend=False)
plt.title(f'Top {top_n_trigrams} Tri-grammes les Plus Fréquents (Après Lemmatisation)')
plt.xlabel('Fréquence')
plt.ylabel('Tri-gramme')
plot_path_trigram = os.path.join(OUTPUT_DIR, '5b_trigram_freq.png')
plt.tight_layout()
plt.savefig(plot_path_trigram)
print(f"Graphique des tri-grammes sauvegardé : {plot_path_trigram}")
# plt.show()

# Commentaire sur la pertinence :
# (À ajouter manuellement après exécution et analyse des graphiques)
# L'analyse des n-grammes peut révéler des expressions ou des séquences de mots
# courantes. Il faut examiner si certaines de ces séquences (ex: "want die",
# "feel like", "not want live", "help me") sont particulièrement associées
# à l'une des classes (suicide / non-suicide). La mesure PMI (Pointwise Mutual Information)
# peut aider à trouver des collocations intéressantes au-delà de la simple fréquence.

# 6. (Optionnel) Vérification de la Loi de Zipf
print("\n[6. Vérification de la Loi de Zipf (Optionnel)]")

# Utiliser la distribution de fréquence des tokens lemmatisés (fdist_lemmatized)
counts = sorted(fdist_lemmatized.values(), reverse=True)
ranks = range(1, len(counts) + 1)

# Créer le graphique log-log
plt.figure(figsize=(10, 6))
plt.plot(np.log(ranks), np.log(counts))
plt.xlabel('Log(Rang)')
plt.ylabel('Log(Fréquence)')
plt.title('Loi de Zipf sur les Tokens Lemmatisés (Log-Log)')
plot_path_zipf = os.path.join(OUTPUT_DIR, '6_zipf_law.png')
plt.grid(True)
plt.savefig(plot_path_zipf)
print(f"Graphique de la loi de Zipf sauvegardé : {plot_path_zipf}")
# plt.show()

# Commentaire sur la loi de Zipf :
# La loi de Zipf stipule que la fréquence d'un mot est inversement proportionnelle
# à son rang dans le classement des fréquences. Sur un graphique log-log, cela
# devrait se traduire par une relation approximativement linéaire avec une pente négative.
# Ce graphique permet de vérifier si la distribution des mots dans notre corpus
# suit cette tendance générale observée dans de nombreuses langues naturelles.

print("\n--- Fin de la Partie 1 ---")
