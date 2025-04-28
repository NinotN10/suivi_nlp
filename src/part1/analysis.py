import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.probability import FreqDist
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.collocations import TrigramAssocMeasures, TrigramCollocationFinder
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import numpy as np

DATASET_PATH = 'dataset/train.parquet'
OUTPUT_DIR = 'outputs/part1/plots'

# Téléchargement des ressources NLTK nécessaires
print("Téléchargement des ressources NLTK (si nécessaire)...")
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
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
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Ressource 'punkt_tab' non trouvée. Téléchargement...")
    nltk.download('punkt_tab')
print("Ressources NLTK vérifiées/téléchargées.")


os.makedirs(OUTPUT_DIR, exist_ok=True)

print("--- Partie 1 : Chargement, Exploration et Traitement ---")

# 1. Chargement et Exploration Initiale
print("\n[1. Chargement et Exploration Initiale]")
try:
    df = pd.read_parquet(DATASET_PATH)
    print(f"Dataset chargé depuis : {DATASET_PATH}")

    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Les colonnes 'text' et 'label' attendues ne sont pas présentes.")
    print("\nColonnes présentes : OK ('text', 'label')")

    # Analyser la distribution des étiquettes (supposant 'suicide' et 'non-suicide')
    print("\nDistribution des étiquettes ('label') :")
    label_distribution = df['label'].value_counts(normalize=True) * 100
    print(label_distribution)
    print(f"\nNombre total d'échantillons : {len(df)}")

    plt.figure(figsize=(6, 4))
    sns.countplot(x='label', data=df, hue='label', palette='viridis', legend=False, order=['non-suicide', 'suicide'])
    plt.title('Distribution des Étiquettes (Suicide vs Non-Suicide)')
    plt.xlabel('Étiquette')
    plt.ylabel('Nombre d\'occurrences')
    plot_path = os.path.join(OUTPUT_DIR, '1_label_distribution.png')
    plt.savefig(plot_path)
    print(f"Graphique de distribution sauvegardé : {plot_path}")

except FileNotFoundError:
    print(f"ERREUR : Le fichier dataset '{DATASET_PATH}' n'a pas été trouvé.")
    exit()
except Exception as e:
    print(f"Une erreur est survenue lors du chargement ou de l'exploration initiale : {e}")
    exit()

# 2. Analyse Fréquentielle sur Données Pré-nettoyées
print("\n[2. Analyse Fréquentielle sur Données Pré-nettoyées]")

# Tokenisation de tous les textes
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

# 3. Traitement Linguistique Complémentaire : Lemmatisation
print("\n[3. Traitement Linguistique Complémentaire : Lemmatisation]")

lemmatizer = WordNetLemmatizer()

# Fonction pour obtenir le POS tag compatible avec WordNetLemmatizer
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# Fonction pour lemmatiser un texte (tokenisé)
def lemmatize_text(text):
    tokens = word_tokenize(text)
    # Lemmatiser chaque token en utilisant son POS tag
    lemmatized_tokens = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokens]
    return lemmatized_tokens

print("Application de la lemmatisation sur la colonne 'text'...")
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
sns.barplot(x=bigram_counts, y=bigram_labels, hue=bigram_labels, palette='mako', legend=False)
plt.title(f'Top {top_n_bigrams} Bi-grammes les Plus Fréquents (Après Lemmatisation)')
plt.xlabel('Fréquence')
plt.ylabel('Bi-gramme')
plot_path_bigram = os.path.join(OUTPUT_DIR, '5a_bigram_freq.png')
plt.tight_layout() # Ajuster pour éviter que les labels se chevauchent
plt.savefig(plot_path_bigram)
print(f"Graphique des bi-grammes sauvegardé : {plot_path_bigram}")

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
sns.barplot(x=trigram_counts, y=trigram_labels, hue=trigram_labels, palette='rocket', legend=False)
plt.title(f'Top {top_n_trigrams} Tri-grammes les Plus Fréquents (Après Lemmatisation)')
plt.xlabel('Fréquence')
plt.ylabel('Tri-gramme')
plot_path_trigram = os.path.join(OUTPUT_DIR, '5b_trigram_freq.png')
plt.tight_layout()
plt.savefig(plot_path_trigram)
print(f"Graphique des tri-grammes sauvegardé : {plot_path_trigram}")

print("\n[6. Vérification de la Loi de Zipf]")

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
print("\n--- Fin de la Partie 1 ---")
