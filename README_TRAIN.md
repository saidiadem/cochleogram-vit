# Guide d'exécution — Classification des sons respiratoires

## Prérequis
- Python 3.10+
- GPU NVIDIA avec support CUDA (recommandé)

---

## Installation

### 1. Créer et activer un environnement virtuel

```bash
python -m venv .venv
```

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

---

### 2. Installer PyTorch

Vérifier votre version CUDA :
```bash
nvidia-smi
```

Repérer `CUDA Version: XX.X` dans la sortie, puis exécuter la commande correspondante :

| Version CUDA | Commande |
|---|---|
| 11.8 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118` |
| 12.1 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
| 12.4 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` |
| CPU uniquement | `pip install torch torchvision` |

> Pour d'autres versions CUDA : https://pytorch.org/get-started/locally/

---

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

### 4. Installer le projet

```bash
pip install -e .
```

---

## Exécution

### Étape 1 — Générer les cochléogrammes RGB
```bash
python scripts/precompute_rgb.py
```

### Étape 2 — Lancer l'entraînement
```bash
python scripts/train.py --config configs/default.yaml
```

---

## Résultats attendus

À la fin de l'entraînement, les métriques suivantes seront affichées pour chaque pli et en agrégé :
- Sensibilité, Spécificité, Précision, Accuracy, Score
- Matrice de confusion binaire (Normal vs Adventice)
- Métriques par classe (Normal, Crackles, Wheezes, Both)

---

## Notes
- Les cochléogrammes bruts sont déjà inclus dans l'archive
- Aucune étape de prétraitement audio n'est nécessaire
