# Lecture de montant sur facture - Computer Vision

Trois méthodes mises en concurrence sur un même jeu de **~50 tickets de caisse réels** : traitement d'image classique, modèle de deep learning pré-entraîné, et modèle de langage multimodal. Qui lit le mieux le montant total ?

![Image de départ](./assets/sample.jpg)

## Le classement

Précision = part des montants exactement retrouvés sur les ~50 factures.

| Approche | Technologie | Précision | MAE | RMSE | R² |
|---|---|:---:|:---:|:---:|:---:|
| Traitement d'image | OpenCV + PyTesseract | très faible | – | – | – |
| Deep learning pré-entraîné | LayoutLMv3 (Hugging Face) | 62 % | 55,97 € | 247,53 € | -1,62 |
| **Langage multimodal** | **Gemma 3 (4B) via Ollama** | **92 %** | **0,26 €** | **1,20 €** | **1,00** |

Le gagnant est net : `gemma3:4b` retrouve 92 % des montants avec une erreur moyenne inférieure à 30 centimes. L'OCR classique reste trop fragile face à la diversité des tickets, et LayoutLMv3 tombe juste ~60 % du temps mais se fait piéger par quelques factures aberrantes (en les écartant : MAE 7,87 € / RMSE 22,00 € / R² 0,78).

> *MAE : erreur absolue moyenne · RMSE : erreur quadratique moyenne · R² : coefficient de détermination (proche de 1 = excellente corrélation).*

## Les trois approches

**1. OpenCV + PyTesseract** — détection des contours, isolation du plus grand, recadrage/redressement, puis OCR du montant.

| ![Contours](./assets/pleincontours.png) | ![Filtrage](./assets/AVECFILTRE.png) | ![Prix détecté](./assets/CASEDEVECFILTRE.png) |
| :---: | :---: | :---: |
| Détection des contours | Filtrage | Prix détecté |

**2. LayoutLMv3** — modèle Hugging Face [pré-entraîné à la détection de factures](https://huggingface.co/Theivaprakasham/layoutlmv3-finetuned-invoice) ([dépôt](https://github.com/Theivaprakasham/layoutlmv3)).

![Résultat Hugging Face](./assets/output.png)

**3. Gemma 3 via Ollama** — le modèle multimodal `gemma3:4b` est interrogé directement sur l'image (un GPU est recommandé).

## Installation

```bash
pip install -r requirements.txt
```

Selon l'approche testée :
- **PyTesseract** requiert le moteur [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract) installé sur la machine.
- **Ollama** doit être installé avec le modèle téléchargé : `ollama pull gemma3:4b` (GPU recommandé).

## Utilisation

Tout est dans le notebook [`Projet_Capture_Facture.ipynb`](./Projet_Capture_Facture.ipynb) : code des trois approches, méthodologie d'évaluation et résultats détaillés. Les utilitaires OpenCV sont dans `imgutils.py`, le jeu de factures dans `data/`.

## Auteurs

Baptiste Behr · Thomas Kusnierek · Benjamin Szurek — projet académique (école d'ingénieurs, 2024-2025).
