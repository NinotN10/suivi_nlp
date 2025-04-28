import polars as pl
import pandas as pd
import spacy
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import os


DATASET_PATH = "dataset/train.parquet"
OUTPUT_DIR = "outputs/part2/plots"
SPACY_MODEL = "en_core_web_sm"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Chargement des données et du modèle ---
print("--- Partie 2 : Analyse Syntaxique et Identification d'Entités ---")

print(f"\n[1. Chargement des données avec Polars et du modèle spaCy ({SPACY_MODEL})]")
try:
    # Charger les données avec Polars depuis le fichier Parquet
    df = pl.read_parquet(DATASET_PATH)
    print(f"Dataset chargé ({len(df)} lignes) depuis {DATASET_PATH}.")
except FileNotFoundError:
    print(f"Erreur : Le fichier {DATASET_PATH} n'a pas été trouvé.")
    exit()

# --- Activation du GPU (si disponible) ---
gpu_activated = False
try:
    # Tente d'activer le GPU. Nécessite spacy[cuda] ou spacy[cudaXXX] installé.
    spacy.require_gpu()
    gpu_activated = True
    print("GPU activé pour spaCy.")
except Exception as e:
    print(f"GPU non activé pour spaCy (raison : {e}). Utilisation du CPU.")
    # Le script continuera sur CPU si le GPU n'est pas disponible/configuré

try:
    nlp = spacy.load(SPACY_MODEL)
    print(f"Modèle spaCy '{SPACY_MODEL}' chargé.")
    if gpu_activated:
        print("Le modèle utilisera le GPU.")
    else:
        print("Le modèle utilisera le CPU.")
except OSError:
    print(f"Erreur : Le modèle spaCy '{SPACY_MODEL}' n'est pas installé.")
    print(f"Veuillez l'installer avec : python -m spacy download {SPACY_MODEL}")
    exit()

# --- 2. Étiquetage Morpho-syntaxique (POS Tagging) ---
print("\n[2. Étiquetage Morpho-syntaxique (POS Tagging)]")

# Fonction pour extraire les POS tags d'un texte
def get_pos_tags(text):
    doc = nlp(str(text)) # Assurer que le texte est une chaîne
    return [token.pos_ for token in doc if not token.is_punct and not token.is_space]

# --- Utilisation de nlp.pipe() pour accélérer le traitement (CPU ou GPU) ---
BATCH_SIZE = 1000 # Augmenter la taille du lot pour le GPU peut être bénéfique
print(f"Utilisation de nlp.pipe() avec batch_size={BATCH_SIZE}")

# Extraire les textes pour nlp.pipe()
texts_to_process = df['text'].cast(pl.Utf8).to_list()

# Appliquer le POS Tagging avec nlp.pipe()
print("Application du POS Tagging...")
pos_tags_list = []
# Utiliser nlp.pipe pour traiter les textes par lots
for doc in nlp.pipe(texts_to_process, batch_size=BATCH_SIZE):
    pos_tags_list.append([token.pos_ for token in doc if not token.is_punct and not token.is_space])

df = df.with_columns(pl.Series("pos_tags", pos_tags_list))
print("POS Tagging terminé.")


print("Agrégation des POS tags par label...")
pos_tags_non_suicide = df.filter(pl.col("label") == "non-suicide").select("pos_tags").explode("pos_tags").to_series().to_list()
pos_tags_suicide = df.filter(pl.col("label") == "suicide").select("pos_tags").explode("pos_tags").to_series().to_list()

pos_counts_non_suicide = Counter(pos_tags_non_suicide)
pos_counts_suicide = Counter(pos_tags_suicide)
print("Agrégation terminée.")

total_tokens_non_suicide = len(pos_tags_non_suicide)
total_tokens_suicide = len(pos_tags_suicide)

pos_df_non_suicide = pd.DataFrame.from_dict(pos_counts_non_suicide, orient='index', columns=['count'])
pos_df_non_suicide['label'] = 'non-suicide'
pos_df_non_suicide['proportion'] = pos_df_non_suicide['count'] / total_tokens_non_suicide if total_tokens_non_suicide > 0 else 0

pos_df_suicide = pd.DataFrame.from_dict(pos_counts_suicide, orient='index', columns=['count'])
pos_df_suicide['label'] = 'suicide'
pos_df_suicide['proportion'] = pos_df_suicide['count'] / total_tokens_suicide if total_tokens_suicide > 0 else 0

pos_df_combined = pd.concat([pos_df_non_suicide, pos_df_suicide]).reset_index().rename(columns={'index': 'pos_tag'})

# Visualisation
print("Génération du graphique de distribution des POS tags...")
plt.figure(figsize=(12, 8))
sns.barplot(data=pos_df_combined, x='pos_tag', y='proportion', hue='label', palette='viridis', hue_order=['non-suicide', 'suicide'])
plt.title('Distribution Proportionnelle des POS Tags par Classe (Non-Suicide vs Suicide)')
plt.xlabel('Catégorie Grammaticale (POS Tag)')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
pos_plot_path = os.path.join(OUTPUT_DIR, "1_pos_distribution.png")
plt.savefig(pos_plot_path)
print(f"Graphique sauvegardé : {pos_plot_path}")
plt.close()

# --- 3. Reconnaissance d'Entités Nommées (NER) ---
print("\n[3. Reconnaissance d'Entités Nommées (NER)]")

def get_ner_labels(text):
    doc = nlp(str(text))
    return [ent.label_ for ent in doc.ents]

# Appliquer la NER avec nlp.pipe()
print("Application de la NER...")
ner_labels_list = []
for doc in nlp.pipe(texts_to_process, batch_size=BATCH_SIZE):
    ner_labels_list.append([ent.label_ for ent in doc.ents])

# Ajouter les résultats comme une nouvelle colonne
df = df.with_columns(pl.Series("ner_labels", ner_labels_list))
print("NER terminée.")


# Agréger les comptes par label en utilisant Polars puis Counter (avec les labels textuels)
print("Agrégation des labels NER par label...")
ner_labels_non_suicide = df.filter(pl.col("label") == "non-suicide").select("ner_labels").explode("ner_labels").to_series().to_list()
ner_labels_suicide = df.filter(pl.col("label") == "suicide").select("ner_labels").explode("ner_labels").to_series().to_list()

ner_counts_non_suicide = Counter(ner_labels_non_suicide)
ner_counts_suicide = Counter(ner_labels_suicide)
print("Agrégation terminée.")

# Convertir en DataFrame pandas pour la visualisation
total_ents_non_suicide = len(ner_labels_non_suicide)
total_ents_suicide = len(ner_labels_suicide)

ner_df_non_suicide = pd.DataFrame.from_dict(ner_counts_non_suicide, orient='index', columns=['count'])
ner_df_non_suicide['label'] = 'non-suicide'
ner_df_non_suicide['proportion'] = ner_df_non_suicide['count'] / total_ents_non_suicide if total_ents_non_suicide > 0 else 0

ner_df_suicide = pd.DataFrame.from_dict(ner_counts_suicide, orient='index', columns=['count'])
ner_df_suicide['label'] = 'suicide'
ner_df_suicide['proportion'] = ner_df_suicide['count'] / total_ents_suicide if total_ents_suicide > 0 else 0

# Filtrer les entités rares si nécessaire pour une meilleure visualisation
ner_df_combined = pd.concat([ner_df_non_suicide, ner_df_suicide]).reset_index().rename(columns={'index': 'ner_label'})
common_ner_labels = ner_df_combined.groupby('ner_label')['label'].nunique()
ner_df_filtered = ner_df_combined[ner_df_combined['ner_label'].isin(common_ner_labels[common_ner_labels > 0].index)]

# Visualisation
if not ner_df_filtered.empty:
    print("Génération du graphique de distribution des types d'entités (NER)...")
    plt.figure(figsize=(12, 8))
    sns.barplot(data=ner_df_filtered, x='ner_label', y='proportion', hue='label', palette='magma', hue_order=['non-suicide', 'suicide'])
    plt.title('Distribution Proportionnelle des Types d\'Entités (NER) par Classe (Non-Suicide vs Suicide)')
    plt.xlabel('Type d\'Entité (NER Label)')
    plt.ylabel('Proportion')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    ner_plot_path = os.path.join(OUTPUT_DIR, "2_ner_distribution.png")
    plt.savefig(ner_plot_path)
    print(f"Graphique sauvegardé : {ner_plot_path}")
    plt.close()
else:
    print("Aucune entité nommée commune trouvée ou détectée pour la visualisation.")

print("\n[4. Analyse des Dépendances Syntaxiques (Commentaires)]")
