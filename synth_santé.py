# Import des packages nécessaires
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# -------------------------------
# Étape 1 : Chargement et prétraitement des données
# -------------------------------

# Charger le dataset (modifiez le chemin selon votre fichier)
df = pd.read_csv("Follow-up_Records.csv")

# Séparer les colonnes numériques et catégorielles
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

# Encoder les colonnes catégorielles en one-hot
encoder = OneHotEncoder(sparse_output=False)
cat_encoded = encoder.fit_transform(df[cat_cols])

# Normaliser les colonnes numériques entre -1 et 1 (pour Tanh)
scaler = MinMaxScaler(feature_range=(-1, 1))
num_scaled = scaler.fit_transform(df[num_cols])

# Combiner les données numériques et catégorielles encodées
data_processed = np.hstack((num_scaled, cat_encoded))

# -------------------------------
# Étape 2 : Définition de l'architecture GAN
# -------------------------------

data_dim = data_processed.shape[1]  # nombre total de features
latent_dim = 64  # dimension du vecteur bruit

# Générateur
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, data_dim),
            nn.Tanh()  # sortie entre -1 et 1
        )
    def forward(self, z):
        return self.model(z)

# Discriminateur
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(data_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()  # probabilité que l'entrée soit réelle
        )
    def forward(self, x):
        return self.model(x)

# -------------------------------
# Étape 3 : Préparation des données pour PyTorch
# -------------------------------

# Conversion en tenseurs PyTorch
real_data = torch.tensor(data_processed, dtype=torch.float32)
dataset = TensorDataset(real_data)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialisation des modèles
generator = Generator()
discriminator = Discriminator()

# Optimiseurs
lr = 0.0002
optim_G = torch.optim.Adam(generator.parameters(), lr=lr)
optim_D = torch.optim.Adam(discriminator.parameters(), lr=lr)

# Fonction de perte
criterion = nn.BCELoss()

# -------------------------------
# Étape 4 : Boucle d'entraînement
# -------------------------------

epochs = 2000
for epoch in range(epochs):
    for real_batch, in loader:
        batch_size = real_batch.size(0)

        # Labels pour vrai et faux
        real_labels = torch.ones((batch_size, 1))
        fake_labels = torch.zeros((batch_size, 1))

        # --- Entraînement du discriminateur ---
        z = torch.randn(batch_size, latent_dim)
        fake_data = generator(z)

        real_loss = criterion(discriminator(real_batch), real_labels)
        fake_loss = criterion(discriminator(fake_data.detach()), fake_labels)
        d_loss = (real_loss + fake_loss) / 2

        optim_D.zero_grad()
        d_loss.backward()
        optim_D.step()

        # --- Entraînement du générateur ---
        z = torch.randn(batch_size, latent_dim)
        fake_data = generator(z)
        g_loss = criterion(discriminator(fake_data), real_labels)  # veut tromper le discriminateur

        optim_G.zero_grad()
        g_loss.backward()
        optim_G.step()

    # Affichage périodique
    if epoch % 200 == 0:
        print(f"Epoch [{epoch}/{epochs}]  D_loss: {d_loss.item():.4f}  G_loss: {g_loss.item():.4f}")

# -------------------------------
# Étape 5 : Génération de dossiers médicaux synthétiques
# -------------------------------

# Générer 10 nouveaux dossiers synthétiques
z = torch.randn(10, latent_dim)
synthetic_data_scaled = generator(z).detach().numpy()

# Inverser la normalisation des données numériques
num_synthetic = scaler.inverse_transform(synthetic_data_scaled[:, :len(num_cols)])

# Inverser l'encodage one-hot des colonnes catégorielles
cat_synthetic = encoder.inverse_transform(synthetic_data_scaled[:, len(num_cols):])

# Créer un DataFrame avec les données synthétiques
synthetic_df = pd.DataFrame(num_synthetic, columns=num_cols)
synthetic_df[cat_cols] = cat_synthetic

print(synthetic_df)
