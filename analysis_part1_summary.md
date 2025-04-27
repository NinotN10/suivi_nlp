# Analyse des Résultats - Partie 1 : Chargement, Exploration et Traitement

Ce document résume les résultats de l'exécution du script `src/part1/analysis.py`, en intégrant l'analyse du texte et une description du contenu des graphiques générés.

## 1. Chargement et Exploration Initiale

*   **Source des données :** `dataset/dataset.csv`
*   **Structure :** 1000 lignes, 2 colonnes (`text`, `label`). Pas de valeurs manquantes.
*   **Distribution des étiquettes :** Très équilibrée (Label 1: 50.1%, Label 0: 49.9%).
*   **Graphique associé :** `outputs/part1/plots/1_label_distribution.png`
    *   *Description :* Ce graphique (probablement un diagramme en barres ou circulaire) visualise la distribution quasi égale des deux étiquettes dans le jeu de données.

## 2. Analyse Fréquentielle (Avant Nettoyage Approfondi)

*   **Tokens bruts :** 65 942
*   **Tokens les plus fréquents (Top 30) :** `not` (3031), `like` (849), `want` (782), `know` (693), `feel` (644), `life` (584), etc.
*   **Graphiques associés :**
    *   `outputs/part1/plots/2a_freq_dist_bar.png`
        *   *Description :* Un diagramme en barres montrant la fréquence des 30 tokens les plus courants avant lemmatisation. La barre pour `not` est la plus haute, suivie par `like`, `want`, etc., illustrant leur prédominance.
    *   `outputs/part1/plots/2b_freq_dist_wordcloud.png`
        *   *Description :* Un nuage de mots où la taille de chaque mot est proportionnelle à sa fréquence brute. Les mots comme `not`, `like`, `want` apparaissent en plus grand, offrant une vue d'ensemble visuelle des termes dominants.

## 3. Traitement Linguistique (Lemmatisation)

*   **Action :** Les mots de la colonne `text` ont été réduits à leur forme de base (lemme). Ex: `going` -> `go`.
*   **Objectif :** Regrouper les différentes formes grammaticales d'un même mot pour une analyse plus précise.

## 4. Analyse Fréquentielle (Après Lemmatisation)

*   **Tokens lemmatisés :** 65 942 (le nombre total reste identique, mais les mots sont sous leur forme lemme).
*   **Lemmes les plus fréquents (Top 30) :** `not` (3031), `want` (914), `go` (877), `like` (870), `get` (868), `feel` (790), `know` (735), etc. Notez l'augmentation de fréquence pour des lemmes comme `want`, `go`, `get`, `feel`, `make` par rapport à leurs formes brutes.
*   **Graphique associé :** `outputs/part1/plots/4_freq_dist_bar_lemmatized.png`
    *   *Description :* Similaire au graphique 2a, mais ce diagramme en barres montre les fréquences des 30 *lemmes* les plus courants. On peut y observer l'effet du regroupement des formes de mots (par exemple, la barre pour `go` est plus haute que dans le graphique 2a car elle inclut `going`).

## 5. Analyse des N-grammes

*   **Objectif :** Identifier les séquences de mots fréquentes pour capturer plus de contexte.
*   **Bi-grammes (Top 20) :** `('safety', 'net')`, `('steam', 'trading')`, `('box', 'cutter')`, `('psych', 'ward')`, `('slit', 'wrist')`, etc.
    *   **Graphique associé :** `outputs/part1/plots/5a_bigram_freq.png`
        *   *Description :* Un diagramme en barres illustrant la fréquence des 20 paires de mots (bi-grammes) les plus courantes, mettant en évidence les collocations fréquentes.
*   **Tri-grammes (Top 20) :** `('steam', 'trading', 'card')`, `('blah', 'blah', 'blah')`, `('low', 'self', 'esteem')`, `('rage', 'rage', 'die')`, etc.
    *   **Graphique associé :** `outputs/part1/plots/5b_trigram_freq.png`
        *   *Description :* Un diagramme en barres illustrant la fréquence des 20 triplets de mots (tri-grammes) les plus courants, révélant des expressions ou thèmes plus spécifiques.

## 6. Vérification de la Loi de Zipf

*   **Graphique associé :** `outputs/part1/plots/6_zipf_law.png`
    *   *Description :* Ce graphique représente le rang d'un mot (par fréquence décroissante) en fonction de sa fréquence, généralement sur une échelle log-log. Si les points forment approximativement une ligne droite descendante, cela indique que la distribution des mots dans le corpus suit la loi de Zipf, ce qui est typique des langues naturelles.

## Conclusion de la Partie 1

L'analyse initiale a révélé un jeu de données équilibré. Les analyses de fréquence (avant/après lemmatisation) et de n-grammes ont identifié les termes et expressions clés. Les visualisations fournissent une compréhension graphique de la distribution des étiquettes, de la fréquence des mots/lemmes, des séquences de mots courantes et de la conformité à la loi de Zipf. Ces informations sont cruciales pour les étapes suivantes d'analyse et de modélisation.
