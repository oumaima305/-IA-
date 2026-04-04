# Rapport d'Analyse Statistique et Machine Learning en Finance

## HAMDOUNE Oumaima 
![Description de l'image](https://raw.githubusercontent.com/USERNAME/REPO/main/HAMDOUNE.png)
## KAISS Salma  
## CAC G2



## Sommaire

1. Introduction
2. [Partie 1 — Statistiques et Loi Normale en Finance](#partie-1)
   - 1.1. Statistiques Descriptives des Portefeuilles
   - 1.2. Visualisation des Distributions
   - 1.3. Test de Normalité et VaR
3. [Partie 2 — Théorème de Bayes Appliqué au Risque de Crédit](#partie-2)
   - 2.1. Calcul de Bayes Manuel
   - 2.2. Mise à Jour Séquentielle
   - 2.3. Fonction Générique Bayes
   - 2.4. Matrice de Confusion et Lien avec Bayes
4. [Partie 3 — Machine Learning : Détection de Défauts de Crédit (KNN)](#partie-3)
   - 3.1. Génération et Exploration du Dataset
   - 3.2. Préprocessing et Split Train/Test
   - 3.3. Modèle KNN et Sélection de K Optimal
   - 3.4. Métriques de Classification
   - 3.5. Courbe ROC
   - 3.6. Calcul du ROI et Recommandation Business
5. [Recommandations](#recommandations)
6. [Conclusion](#conclusion)

---

## 1. Introduction

Le présent rapport vise à documenter et analyser un travail pratique (TP) portant sur l'application des méthodes statistiques et du machine learning dans le domaine de la finance, et plus particulièrement dans la gestion du risque de portefeuille et la détection de défauts de crédit.

Ce TP est structuré en trois grandes parties :

- **Partie 1** traite des statistiques descriptives, de la loi normale et du calcul de la Value at Risk (VaR) à partir de rendements historiques de deux portefeuilles d'investissement (conservateur et agressif).
- **Partie 2** applique le théorème de Bayes à l'évaluation du risque de crédit, en mettant à jour séquentiellement les probabilités de défaut à la lumière de nouveaux événements observés.
- **Partie 3** met en œuvre un modèle de classification par K plus proches voisins (KNN) pour prédire les défauts de crédit, en évaluant ses performances à l'aide de métriques standards et d'une analyse de retour sur investissement (ROI).

L'ensemble des analyses est réalisé en Python, en exploitant des bibliothèques scientifiques telles que NumPy, Pandas, Matplotlib, Seaborn, SciPy et scikit-learn.

---

## 2. Partie 1 — Statistiques et Loi Normale en Finance

### 2.1. Statistiques Descriptives des Portefeuilles

#### Explication du code

La fonction `calculer_stats_portefeuille(rendements, nom)` est définie pour centraliser le calcul de toutes les statistiques descriptives d'un portefeuille. Elle prend en entrée un tableau NumPy de rendements mensuels et le nom du portefeuille, puis retourne un dictionnaire contenant les indicateurs clés suivants :

- **Moyenne mensuelle** (`np.mean`) : mesure du rendement moyen sur la période.
- **Écart-type mensuel** (`np.std(ddof=1)`) : mesure de la dispersion des rendements. L'utilisation de `ddof=1` assure une estimation non biaisée pour un échantillon.
- **Médiane** (`np.median`) : indicateur de tendance centrale, résistant aux valeurs extrêmes.
- **Rendement annualisé** : calculé par capitalisation composée selon la formule `((1 + r_mensuel/100)^12 - 1) × 100`, qui capture l'effet des intérêts composés sur une année.
- **Volatilité annualisée** : obtenue en multipliant l'écart-type mensuel par `√12`, conformément à la règle de la racine du temps (valide sous hypothèse d'indépendance des rendements).

Les données initiales correspondent à 24 mois de rendements historiques pour deux portefeuilles :

- **Portefeuille A (Conservateur)** : rendements stables et positifs, fluctuant entre –0,5% et +1,5%.
- **Portefeuille B (Agressif)** : rendements plus volatils, allant de –4,2% à +8,5%, reflétant une exposition plus élevée au risque de marché.

#### Résultats obtenus

| Indicateur | Portefeuille A (Conservateur) | Portefeuille B (Agressif) |
|---|---|---|
| Rendement mensuel moyen | 0,94% | 2,89% |
| Écart-type mensuel | 0,48% | 4,45% |
| Médiane | 1,00% | 4,70% |
| Rendement annualisé | 11,85% | 40,79% |
| Volatilité annualisée | 1,65% | 15,41% |

#### Interprétation

Ces résultats illustrent le compromis fondamental rendement/risque en finance. Le portefeuille B présente un rendement annualisé nettement supérieur (40,79% contre 11,85%), mais sa volatilité annualisée est presque dix fois plus élevée (15,41% contre 1,65%). La médiane du portefeuille B (4,70%) est supérieure à sa moyenne (2,89%), suggérant la présence de rendements négatifs importants qui tirent la moyenne vers le bas.

Le portefeuille A, quant à lui, affiche une grande régularité avec une médiane proche de la moyenne, ce qui témoigne d'une distribution symétrique et peu risquée.

---

### 2.2. Visualisation des Distributions

#### Explication du code

Le code crée deux graphiques côte à côte via `plt.subplots(1, 2)` :

1. **Histogrammes superposés** (`ax1.hist`) : les rendements des deux portefeuilles sont représentés en densité de probabilité (`density=True`), avec des couleurs distinctes (vert pour A, rouge pour B) et une transparence (`alpha=0.6`) pour lisibilité. Des lignes verticales en pointillés (`axvline`) marquent les moyennes respectives.

2. **Boxplots comparatifs** (`ax2.boxplot`) : ce type de graphique résume visuellement la distribution en montrant la médiane, les quartiles (Q1, Q3) et les valeurs aberrantes. L'option `patch_artist=True` permet de colorier les boîtes.

#### Interprétation des graphiques

L'histogramme révèle clairement la différence de profil entre les deux portefeuilles. La distribution du portefeuille A est étroite et centrée autour de 1%, confirmant sa faible volatilité. En revanche, la distribution du portefeuille B est étalée sur un large intervalle (de –4% à +9%), démontrant sa nature agressive avec des potentiels de gains et de pertes importants.

Les boxplots quantifient ces différences : la boîte du portefeuille B est beaucoup plus large (IQR élevé), et l'on observe des valeurs extrêmes côté négatif, indicatrices de mois particulièrement défavorables. L'axe horizontal à zéro (`axhline(0)`) permet de visuellement identifier les périodes de rendements négatifs pour chaque portefeuille.

---

### 2.3. Test de Normalité et Value at Risk (VaR)

#### Explication du code

Le calcul de la VaR s'appuie sur l'hypothèse de normalité des rendements. La démarche suit deux étapes :

1. **Test de Shapiro-Wilk** (`stats.shapiro`) : ce test statistique évalue si les données suivent une loi normale. Un p-value > 0,05 indique qu'on ne peut pas rejeter l'hypothèse de normalité.

2. **Calcul de la VaR** : pour un niveau de confiance de 95%, la VaR est calculée par :
   - `VaR = μ - z × σ`, où z = 1,645 (quantile à 5% de la loi normale standard)
   - La VaR monétaire représente le montant de perte maximale sur le capital investi (500 000 €) pour une probabilité de dépassement de 5%.

La VaR permet de répondre à la question : *"Quelle est la perte maximale que je risque de subir dans 95% des cas ?"*

#### Résultats et interprétation

Pour un capital de 500 000 €, le calcul de la VaR mensuelle à 95% montre que :
- Le portefeuille A présente une VaR bien inférieure à 50 000 € (seuil de tolérance défini), confirmant sa compatibilité avec les contraintes de risque imposées.
- Le portefeuille B dépasse fréquemment ce seuil, ce qui constitue un signal d'alerte important pour un investisseur dont la perte maximale tolérée est fixée à 10% du capital.

---

## 3. Partie 2 — Théorème de Bayes Appliqué au Risque de Crédit

### 3.1. Contexte

La partie 2 modélise le processus de mise à jour du risque de crédit d'un client à mesure que de nouvelles informations comportementales deviennent disponibles. Le cadre bayésien est particulièrement adapté à ce type de problème, car il permet d'intégrer séquentiellement des évidences pour affiner une estimation probabiliste.

Le contexte est le suivant : un client appartenant au **Segment Standard** (taux de défaut a priori de 5%) présente successivement des événements de risque. Les paramètres utilisés sont :

| Événement | P(E\|Défaut) | P(E\|Non-Défaut) |
|---|---|---|
| Retard de paiement | 80% | 10% |
| Découvert > 500€ | 65% | 15% |
| Refus de crédit ailleurs | 70% | 5% |

---

### 3.2. Question 2.1 — Calcul de Bayes Manuel

#### Explication du code

Le code implémente directement le théorème de Bayes à deux étapes :

**Étape 1 — Calcul de P(Retard) par la loi des probabilités totales :**
```
P(Retard) = P(Retard|Défaut) × P(Défaut) + P(Retard|Non-Défaut) × P(Non-Défaut)
P(Retard) = 0,80 × 0,05 + 0,10 × 0,95 = 0,0400 + 0,0950 = 0,1350
```

**Étape 2 — Application du théorème de Bayes :**
```
P(Défaut|Retard) = P(Retard|Défaut) × P(Défaut) / P(Retard)
P(Défaut|Retard) = (0,80 × 0,05) / 0,1350 = 0,0400 / 0,1350 ≈ 29,63%
```

#### Résultats et interprétation

| Indicateur | Valeur |
|---|---|
| Prior P(Défaut) | 5,0% |
| Posterior P(Défaut\|Retard) | 29,63% |
| Augmentation du risque | +24,6 points |
| Facteur multiplicatif | ×5,93 |

L'observation d'un simple retard de paiement multiplie le risque de défaut par près de 6. Ce résultat illustre le pouvoir informatif de cet événement : un client qui présente un retard a une probabilité de défaut de près de 30%, contre seulement 5% en l'absence d'information. La décision métier recommandée est une **Surveillance Renforcée** (monitoring hebdomadaire, réduction du plafond de découvert de 30%).

---

### 3.3. Question 2.2 — Mise à Jour Séquentielle

#### Explication du code

La mise à jour séquentielle consiste à utiliser la probabilité a posteriori obtenue en Q2.1 comme nouveau prior. Ce mécanisme est la force du bayésianisme : chaque nouvelle observation vient affiner l'estimation précédente.

Le code applique une seconde fois la formule de Bayes, avec :
- **Nouveau prior** : 0,2963 (posterior de Q2.1)
- **Événement 2** : Découvert > 500€ (`P(E|Défaut) = 0,65`, `P(E|Non-Défaut) = 0,15`)

**Calcul :**
```
P(Découvert) = 0,65 × 0,2963 + 0,15 × 0,7037 ≈ 0,2981
P(Défaut|Retard ET Découvert) = (0,65 × 0,2963) / 0,2981 ≈ 64,60%
```

#### Interprétation du graphique

Le graphique en courbe montre l'évolution de la probabilité de défaut à travers les trois étapes :

- **Étape 0** (Prior initial) : 5,0%
- **Étape 1** (Après Retard de paiement) : 29,63%
- **Étape 2** (Après Découvert > 500€) : 64,60%

Deux lignes horizontales en pointillés représentent les seuils décisionnels :
- **Seuil orange à 15%** : déclenchement d'une surveillance renforcée
- **Seuil rouge à 30%** : déclenchement d'une restriction de crédit

La courbe montre une progression exponentielle du risque. Après deux événements défavorables, la probabilité de défaut a été multipliée par 12,92 par rapport au prior initial. Ce profil extrêmement dégradé justifie le passage immédiat à un régime de restriction crédit dès l'étape 2.

---

### 3.4. Question 2.3 — Fonction Générique Bayes

#### Explication du code

La fonction `bayes_update(prior, likelihood_pos, likelihood_neg)` encapsule la logique bayésienne dans un module réutilisable. Elle inclut :

1. **Validation des entrées** : vérification que les trois paramètres appartiennent à l'intervalle [0, 1], avec levée d'une `ValueError` en cas de violation.
2. **Calcul de P(Evidence)** par la loi des probabilités totales.
3. **Protection contre la division par zéro** : si `p_evidence = 0`, la fonction retourne 0.
4. **Application du théorème de Bayes** pour obtenir la probabilité a posteriori.

#### Test sur un client Segment Risque (prior = 15%)

| Étape | P(Défaut) |
|---|---|
| Prior initial | 15,00% |
| Après Retard | 58,54% |
| Après Découvert | 85,95% |
| Après Refus de crédit | 97,68% |

Ce test illustre à quel point l'accumulation d'événements défavorables chez un client à risque initial élevé conduit à une quasi-certitude de défaut (97,68%). La recommandation est sans ambiguïté : **REJET du crédit ou exigence de garanties renforcées**.

---

### 3.5. Question 2.4 — Matrice de Confusion et Lien avec Bayes

#### Explication du code

Le code relie formellement le théorème de Bayes aux métriques de classification en machine learning. En considérant 10 000 clients testés :

- **TP (Vrais Positifs)** : 400 — défauts correctement détectés via un retard
- **FP (Faux Positifs)** : 950 — non-défauts présentant néanmoins un retard
- **FN (Faux Négatifs)** : 100 — défauts non signalés
- **TN (Vrais Négatifs)** : 8 550 — non-défauts sans retard

La **Précision** est calculée comme :
```
Précision = TP / (TP + FP) = 400 / (400 + 950) = 400 / 1350 ≈ 29,63%
```

#### Interprétation — Lien mathématique Bayes ↔ Précision

Le résultat est remarquable : la Précision issue de la matrice de confusion (29,63%) est identique à la probabilité a posteriori calculée par Bayes en Q2.1 (29,63%). Ce n'est pas une coïncidence — c'est une équivalence mathématique fondamentale :

- Le **Théorème de Bayes** calcule `P(Défaut|Retard)`, c'est-à-dire la proportion de défauts réels parmi les clients présentant un retard.
- La **Précision** (machine learning) mesure `TP / (TP + FP)`, soit la proportion de prédictions positives correctes.

Ces deux formules sont mathématiquement équivalentes. Cette équivalence constitue le fondement théorique du classifieur Naïf Bayésien, qui optimise directement les probabilités a posteriori.

---

## 4. Partie 3 — Machine Learning : Détection de Défauts de Crédit (KNN)

### 4.1. Question 3.1 — Génération et Exploration du Dataset

#### Explication du code

Un dataset synthétique de **2 000 clients** est généré via NumPy avec une graine aléatoire fixée (`np.random.seed(42)`) pour assurer la reproductibilité. Les 8 variables (features) retenues sont :

| Feature | Type | Description |
|---|---|---|
| `age` | Entier | Âge du client (25–65 ans) |
| `salaire` | Flottant | Revenu annuel (loi normale, µ=50 000, σ=20 000) |
| `anciennete_emploi` | Flottant | Durée en poste (loi exponentielle) |
| `dette_totale` | Flottant | Montant total des dettes |
| `ratio_dette_revenu` | Flottant | Ratio d'endettement |
| `nb_credits_actifs` | Entier | Nombre de crédits en cours (loi de Poisson) |
| `historique_retards` | Entier | Nombre de retards passés (loi de Poisson) |
| `score_credit_bureau` | Flottant | Score de crédit (300–850) |

La variable cible `defaut` (0/1) est générée probabilistiquement avec des facteurs de risque cumulatifs, conduisant à un **taux de défaut de 16,7%**, caractéristique d'un portefeuille de crédit légèrement déséquilibré (déséquilibre de classes).

---

### 4.2. Question 3.2 — Préprocessing et Split Train/Test

#### Explication du code

Le préprocessing suit trois étapes essentielles :

1. **Séparation features/cible** : `X = df.drop('defaut', axis=1)` et `y = df['defaut']`. Le dataset comporte 8 features et 2 000 observations.

2. **Split stratifié 70/30** : `train_test_split(..., stratify=y)` garantit que la proportion de défauts est maintenue dans chaque partition, évitant un biais d'échantillonnage. Cela donne 1 400 exemples d'entraînement et 600 de test.

3. **Normalisation par StandardScaler** : chaque feature est transformée en score Z `((x - µ) / σ)`. La normalisation est **ajustée uniquement sur le train set** (`fit_transform`) puis **appliquée au test set** (`transform`), évitant la fuite de données (*data leakage*).

#### Résultats — Distribution des classes

| Classe | Train | Test |
|---|---|---|
| 0 (Non-défaut) | 83,29% | ~83,3% |
| 1 (Défaut) | 16,71% | ~16,7% |

La stratification assure une distribution cohérente entre les deux partitions, condition nécessaire pour l'évaluation non biaisée du modèle.

---

### 4.3. Question 3.3 — Modèle KNN et Sélection de K Optimal

#### Explication du code

L'algorithme K-Nearest Neighbors (KNN) prédit la classe d'un exemple en se basant sur les K voisins les plus proches dans l'espace des features. La sélection du paramètre K optimal se fait par **validation croisée à 5 plis** (`StratifiedKFold`), qui partitionne le train set en 5 sous-ensembles pour évaluer chaque valeur de K sans biais.

La métrique utilisée est l'**AUC-ROC** (Area Under the ROC Curve), préférable à l'accuracy pour les datasets déséquilibrés car elle mesure la capacité discriminante du modèle indépendamment du seuil.

Le code balaye des valeurs de K de 1 à 50 (nombres impairs uniquement, pour éviter les égalités) et sélectionne le K maximisant l'AUC-ROC moyen sur les 5 plis.

---

### 4.4. Question 3.4 — Métriques de Classification

#### Explication du code

Les métriques calculées sont :

- **Accuracy** : proportion d'exemples bien classifiés.
- **Précision** : `TP / (TP + FP)` — parmi les défauts prédits, combien sont réels ?
- **Recall (Sensibilité)** : `TP / (TP + FN)` — parmi les défauts réels, combien ont été détectés ?
- **F1-Score** : moyenne harmonique de la Précision et du Recall, utile en cas de déséquilibre de classes.
- **AUC-ROC** : mesure la capacité globale de discrimination du modèle.

**Résultats pour le seuil par défaut (0,5) :**

| Métrique | Valeur |
|---|---|
| TP | 22 |
| FP | 68 |
| FN | 78 |
| Recall | 22% |
| AUC-ROC | 0,54 |
| ROI Net | –955 600 € |

Ces résultats indiquent de **faibles performances du modèle KNN** sur ce dataset. Un AUC de 0,54 est proche du hasard (AUC = 0,50), suggérant que le modèle a une capacité discriminante très limitée pour détecter les défauts de crédit.

---

### 4.5. Question 3.5 — Courbe ROC

#### Explication du code

La courbe ROC (Receiver Operating Characteristic) trace le Taux de Vrais Positifs (TPR = Recall) en fonction du Taux de Faux Positifs (FPR) pour tous les seuils de décision possibles. Elle est calculée via `roc_curve(y_test, y_pred_proba)` de scikit-learn.

La diagonale en pointillés représente un classifieur aléatoire (AUC = 0,50). Un bon modèle s'arc-boute vers le coin supérieur gauche, avec une AUC proche de 1.

#### Interprétation du graphique

La courbe ROC du modèle KNN optimal se situe légèrement au-dessus de la diagonale, avec une AUC de 0,54. Cette valeur confirme que le modèle KNN n'est pas adapté à ce problème dans sa configuration actuelle. La quasi-absence de courbure indique que le modèle ne parvient pas à distinguer efficacement les défauts des non-défauts.

L'analyse par seuil (0,3 ; 0,5 ; 0,7) révèle que les résultats sont identiques pour ces trois valeurs (TP=22, FP=68, FN=78), ce qui traduit une concentration des probabilités prédites autour de valeurs similaires, symptôme d'un modèle peu discriminant.

---

### 4.6. Question 3.6 — Calcul ROI et Recommandation Business

#### Explication du code

L'analyse financière quantifie l'impact économique du modèle. Les paramètres monétaires sont :

| Paramètre | Valeur |
|---|---|
| Gain par défaut détecté (TP) | +15 000 € |
| Coût d'analyse par fausse alerte (FP) | –500 € |
| Coût d'opportunité par fausse alerte (FP) | –1 200 € |
| Coût total par FP | –1 700 € |
| Perte par défaut manqué (FN) | –15 000 € |

La formule du ROI Net est :
```
ROI Net = (TP × 15 000) – (FP × 1 700) – (FN × 15 000)
ROI Net = (22 × 15 000) – (68 × 1 700) – (78 × 15 000)
ROI Net = 330 000 – 115 600 – 1 170 000 = –955 600 €
```

#### Résultats et interprétation

Pour tous les seuils testés (0,3 ; 0,5 ; 0,7), le ROI net s'élève à **–955 600 €** — un résultat catastrophique d'un point de vue business. La raison principale est le faible Recall (22%) : le modèle manque 78 défauts sur 100, chacun représentant une perte de 15 000 €.

L'analyse confirme qu'**aucun seuil testé ne permet d'atteindre l'objectif business d'un Recall ≥ 80%**, rendant ce modèle inadapté à une utilisation opérationnelle.

**Executive Summary :** Le modèle KNN (K=1 optimal) affiche une AUC-ROC de seulement 0,54, traduisant une faible capacité discriminante. Avec un Recall de 22% et une Précision de 24%, le ROI estimé est de –955 600 €, bien en deçà de la viabilité économique. Le modèle actuel est incapable d'atteindre l'objectif métier d'un Recall de 80%, nécessitant impérativement une révision de la stratégie de modélisation.

---

## 5. Recommandations

### 5.1. Recommandations pour la Gestion de Portefeuille (Partie 1)

Au regard des résultats statistiques obtenus, les recommandations suivantes peuvent être formulées :

**Pour le choix entre les portefeuilles A et B :**
Le portefeuille B offre un rendement annualisé supérieur (40,79% vs 11,85%), mais au prix d'une volatilité annualisée near 10 fois supérieure (15,41% vs 1,65%). Le choix dépend du profil de risque de l'investisseur. Pour un capital de 500 000 € avec une perte maximale tolérée de 50 000 € (10%), le portefeuille A est le seul compatible avec les contraintes de risque, car la VaR mensuelle du portefeuille B dépasse régulièrement ce seuil.

**Pour une optimisation :** Une allocation mixte (par exemple 70% en A, 30% en B) permettrait de bénéficier d'un meilleur rendement tout en maîtrisant la VaR globale, en exploitant les effets de diversification.

### 5.2. Recommandations pour le Scoring Bayésien (Partie 2)

Le cadre bayésien déployé constitue une base solide pour un système de scoring de crédit. Cependant, plusieurs améliorations sont envisageables :

- **Calibration des vraisemblances** : les valeurs de `P(E|Défaut)` et `P(E|Non-Défaut)` devraient être estimées empiriquement sur des données historiques réelles plutôt que définies a priori.
- **Prise en compte de la dépendance entre événements** : le modèle actuel suppose l'indépendance conditionnelle des événements. Dans la réalité, un retard de paiement et un découvert sont souvent corrélés, ce qui tend à surestimer l'actualisation du risque.
- **Intégration dans un système de scoring continu** : la fonction `bayes_update()` peut être intégrée dans un pipeline de décision en temps réel, permettant de mettre à jour automatiquement le score de risque d'un client à chaque nouvel événement.

### 5.3. Recommandations pour le Modèle de Détection (Partie 3)

Les performances du modèle KNN sont insuffisantes pour une application opérationnelle. Les recommandations sont structurées selon trois axes :

**Amélioration des données :**
- Collecter davantage de données, notamment sur les cas de défaut (classe minoritaire) afin de réduire le déséquilibre de classes.
- Appliquer des techniques de rééchantillonnage telles que SMOTE (Synthetic Minority Over-sampling Technique) pour rééquilibrer artificiellement le dataset d'entraînement.

**Amélioration du modèle :**
- Tester des algorithmes plus performants sur des données tabulaires déséquilibrées : Random Forest, Gradient Boosting (XGBoost, LightGBM), ou Régression Logistique avec pénalisation.
- Optimiser les hyperparamètres via une recherche sur grille (*Grid Search*) ou des méthodes bayésiennes d'optimisation.
- Explorer des métriques d'entraînement adaptées aux classes déséquilibrées (F1-score pondéré, coût asymétrique).

**Ajustement du seuil de décision :**
- Plutôt que d'utiliser le seuil par défaut de 0,5, déterminer le seuil optimal par une analyse coût-bénéfice basée sur la courbe ROC, en tenant compte de l'asymétrie des coûts entre FP et FN. Compte tenu des paramètres financiers (coût FN = 15 000 €, coût FP = 1 700 €), un Recall élevé est prioritaire, ce qui suggère d'abaisser le seuil de décision si les probabilités prédites le permettent.

---

## 6. Conclusion

Ce travail pratique a permis d'explorer trois domaines complémentaires de l'analyse quantitative en finance :

**En Partie 1**, l'analyse statistique des rendements de portefeuilles a mis en évidence le compromis fondamental rendement/risque. La loi normale s'est avérée être un modèle acceptable pour les données du portefeuille A, permettant le calcul d'une VaR fiable. La modélisation par la loi normale reste cependant une approximation, car les rendements financiers réels exhibent souvent des queues épaisses (*fat tails*) que la distribution gaussienne ne capture pas.

**En Partie 2**, l'application du théorème de Bayes a démontré sa puissance dans la mise à jour dynamique du risque de crédit. La progression spectaculaire de la probabilité de défaut — de 5% à 64,60% après deux événements défavorables — illustre l'importance de l'information comportementale dans l'évaluation du risque. L'équivalence mathématique entre la Précision (ML) et la probabilité a posteriori (Bayes) constitue un pont conceptuel important entre statistiques bayésiennes et apprentissage automatique.

**En Partie 3**, l'implémentation du modèle KNN a révélé ses limites sur un dataset déséquilibré : avec une AUC de 0,54 et un ROI négatif de –955 600 €, le modèle n'est pas opérationnel dans sa configuration actuelle. Cette conclusion, bien que décevante en termes de performance immédiate, est précieuse : elle guide les décisions futures vers des approches plus adaptées (modèles ensemblistes, rééchantillonnage, optimisation du seuil) et souligne l'importance cruciale du choix de métrique d'évaluation en contexte de classes déséquilibrées.

En synthèse, ce TP illustre la complémentarité des approches statistiques classiques (loi normale, Bayes) et des méthodes de machine learning moderne dans l'analyse du risque financier. La rigueur méthodologique — validation des hypothèses, gestion du déséquilibre de classes, évaluation économique des modèles — est indispensable pour produire des analyses actionnables dans un environnement professionnel.


