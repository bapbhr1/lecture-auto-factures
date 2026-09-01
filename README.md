# Extraction automatique du montant total d'une facture

**Étude comparative de trois approches de vision par ordinateur et d'IA pour lire le montant total d'un ticket de caisse à partir de sa photo.**

Ce projet met en concurrence trois familles de méthodes — traitement d'image classique, modèle de deep learning pré-entraîné, et modèle de langage multimodal — évaluées sur un même jeu de ~50 factures réelles avec les métriques MAE, RMSE et R².

![Image de départ](./assets/sample.jpg)

## Résultats en un coup d'œil

Évaluation sur l'ensemble des ~50 factures (précision = part des montants exactement retrouvés) :

| Approche | Technologies | Précision | MAE | RMSE | R² |
|---|---|:---:|:---:|:---:|:---:|
| Traitement d'image classique | OpenCV + PyTesseract (OCR) | très faible | – | – | – |
| Modèle pré-entraîné | LayoutLMv3 (Hugging Face) | 62 % | 55,97 € | 247,53 € | -1,62 |
| **Modèle de langage multimodal** | **Gemma 3 (4B) via Ollama** | **92 %** | **0,26 €** | **1,20 €** | **1,00** |

> *MAE : erreur absolue moyenne · RMSE : erreur quadratique moyenne · R² : coefficient de détermination. Plus la MAE et la RMSE sont faibles, mieux c'est ; un R² proche de 1 indique une excellente corrélation.*
>
> *Les métriques d'erreur de LayoutLMv3 sont très dégradées par quelques factures aberrantes (montants > 250 €). En les écartant, le modèle atteint MAE 7,87 € / RMSE 22,00 € / R² 0,78.*

**Conclusion :** l'OCR classique reste trop fragile face à la diversité des tickets ; le modèle pré-entraîné LayoutLMv3 retrouve le bon montant dans ~60 % des cas mais reste sensible aux valeurs extrêmes ; le modèle de langage multimodal `gemma3:4b` obtient de loin les meilleures performances, avec une erreur moyenne inférieure à 30 centimes et 92 % de montants exacts.

---

## Détail des approches

### 1. Traitement d'image classique — OpenCV + PyTesseract


Cette première approche repose sur du traitement d'image classique. À partir de l'image de départ, les librairies OpenCV et PyTesseract sont utilisées pour détecter les contours, isoler le plus grand contour, recadrer et redresser l'image, puis extraire le montant de la facture par OCR :

| ![Contours](./assets/pleincontours.png) | ![Filtrage](./assets/AVECFILTRE.png) | ![Prix détecté](./assets/CASEDEVECFILTRE.png) |
| :-------------------------------------: | :----------------------------------: | :---------------------------------------------: |
|         Détection des contours         |               Filtrage               |                 Prix détecté                 |

### 2. Modèle pré-entraîné — LayoutLMv3 (Hugging Face)

Ce modèle a été **pré-entraîné à la détection de factures** ; il provient de la plateforme Hugging Face :

- Modèle : [https://huggingface.co/Theivaprakasham/layoutlmv3-finetuned-invoice](https://huggingface.co/Theivaprakasham/layoutlmv3-finetuned-invoice)
- Dépôt GitHub : [https://github.com/Theivaprakasham/layoutlmv3](https://github.com/Theivaprakasham/layoutlmv3)

Il permet d'obtenir ce type de résultat :

![Résultat Hugging Face](./assets/output.png)

### 3. Modèle de langage multimodal — Gemma 3 via Ollama

Cette dernière approche utilise le modèle `gemma3:4b` servi par **Ollama**, interrogé pour détecter le montant de la facture.

> L'utilisation d'un **GPU** est fortement recommandée pour exécuter Ollama.

---

## Structure du dépôt

```
.
├── Projet_Capture_Facture.ipynb   # Notebook principal : les 3 approches, l'évaluation et les résultats
├── imgutils.py                    # Fonctions utilitaires de traitement d'image (OpenCV)
├── requirements.txt               # Dépendances Python
├── assets/                        # Images d'illustration du README
└── data/                          # Jeu de données de factures utilisé pour l'évaluation
```

## Installation

```bash
pip install -r requirements.txt
```

Prérequis supplémentaires selon les approches :
- **PyTesseract** nécessite le moteur [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract) installé sur la machine.
- **Ollama** doit être installé, avec le modèle `gemma3:4b` téléchargé (`ollama pull gemma3:4b`). Un GPU est recommandé.

## Utilisation

Ouvrir et exécuter le notebook [`Projet_Capture_Facture.ipynb`](./Projet_Capture_Facture.ipynb), qui contient l'ensemble du code, la méthodologie d'évaluation et les résultats détaillés des trois approches.

---

## Auteurs

Baptiste Behr · Thomas Kusnierek · Benjamin Szurek

*Projet réalisé dans un cadre académique (école d'ingénieurs, 2024-2025).*
