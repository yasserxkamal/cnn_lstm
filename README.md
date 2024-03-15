## Installation

1. Clonez le dépôt sur votre machine locale :

```bash
git clone https://git@github.com:yasserxkamal/cnn_lstm.git
```

2. Accédez au répertoire du projet :

```bash
cd scraper-facebook
```

3. Créez un environnement virtuel Python et activez-le :

```bash
python -m venv env
source env/bin/activate  # Sur Windows, utilisez `env\Scripts\activate`
```

4. Installez les dépendances requises :

```bash
pip install -r requirements.txt
```

5. installez les fichiers nécessaires

Téléchargez les fichiers 'train.csv' et 'wiki-news-300d-1M.vec' à partir du lien suivant : [Google Drive](https://drive.google.com/drive/folders/1vQ5bYntHmKjf4yjQrXyIev3lHmaByQOp).
Placez ces fichiers dans le même dossier que le code.
Vous êtes maintenant prêt à utiliser les données

## Utilisation

Une fois que tous les prérequis sont installés et que le fichier `.env` est configuré, vous pouvez exécuter le script avec la commande suivante :

```bash
python CNN_LSTM.py
```
