import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from CAE import CAE
from CVAE import loss_cvae, CVAE
from CAAE import Encoder, Decoder, Discriminator

# Cargar el archivo .npy
datos = np.load('olivetti_faces.npy')

# Consultar sus dimensiones y tipo de dato (muy útil para entender la estructura)
print("Forma original:", datos.shape)
print("Tipo de dato:", datos.dtype)

# Mezclar los índices de forma reproducible antes de dividir (25% test, 75% validación)
rng = np.random.default_rng(42)
indices = rng.permutation(len(datos))
corte = int(len(datos) * 0.25)

test = datos[indices[:corte]]
validacion = datos[indices[corte:]]

print("Forma test (25%):", test.shape)
print("Forma validación (75%):", validacion.shape)


# Convertir de NumPy a Tensor y agregar dimensión de canal -> (N, 1, 64, 64)
x_train = torch.from_numpy(validacion).unsqueeze(1)
x_test = torch.from_numpy(test).unsqueeze(1)

# En un autoencoder la entrada y el objetivo son la misma imagen
train_loader = DataLoader(TensorDataset(x_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(x_test), batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=1e-3)


def train_vca(model):

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 51):
        model.train()
        total_loss = 0.0
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch_x)
            loss = loss_cvae(recon, batch_x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)

        print(f"Época {epoch:02d} | Loss: {total_loss / len(train_loader.dataset):.4f}")

# Generar 5 caras nuevas desde ruido puro (sin imagen de entrada)


def train_from_model(model):
    criterion = nn.MSELoss()
    optimizer = create_optimizer(model)

    
    epochs = 30
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
    
        train_loss /= len(train_loader.dataset)
    
        # Evaluación en test
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for (batch_x,) in test_loader:
                batch_x = batch_x.to(device)
                output = model(batch_x)
                loss = criterion(output, batch_x)
                test_loss += loss.item() * batch_x.size(0)
    
        test_loss /= len(test_loader.dataset)
        print(f"Época {epoch:02d}/{epochs} | Train MSE: {train_loss:.5f} | Test MSE: {test_loss:.5f}")


def train_CAAE(encoder, decoder, discriminator):
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    opt_ae = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4)
    opt_g = torch.optim.Adam(encoder.parameters(), lr=2e-4)
    for epoch in range(1, 51):
        encoder.train()
        decoder.train()
        discriminator.train()

        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            bs = batch_x.size(0)
            real_labels = torch.ones(bs, 1, device=device)
            fake_labels = torch.zeros(bs, 1, device=device)

            # --- 1. Fase de reconstrucción (Encoder + Decoder) ---
            opt_ae.zero_grad()
            z = encoder(batch_x)
            recon = decoder(z)
            loss_recon = mse_loss(recon, batch_x)
            loss_recon.backward()
            opt_ae.step()

            # --- 2. Fase del Discriminador ---
            opt_d.zero_grad()
            z_real = torch.randn(bs, latent_dim, device=device)  # N(0, 1)
            z_fake = encoder(batch_x).detach()  # detach() evita actualizar al encoder aquí

            loss_d_real = bce_loss(discriminator(z_real), real_labels)
            loss_d_fake = bce_loss(discriminator(z_fake), fake_labels)
            loss_d = 0.5 * (loss_d_real + loss_d_fake)
            loss_d.backward()
            opt_d.step()

            # --- 3. Fase del Generador (Encoder intenta engañar a D) ---
            opt_g.zero_grad()
            z_gen = encoder(batch_x)
            loss_g = bce_loss(discriminator(z_gen), real_labels)  # quiere etiqueta 1
            loss_g.backward()
            opt_g.step()

        print(f"Época {epoch:02d} | Recon: {loss_recon.item():.4f} | D: {loss_d.item():.4f} | G: {loss_g.item():.4f}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 32
encoder = Encoder(latent_dim).to(device)
decoder = Decoder(latent_dim).to(device)
discriminator = Discriminator(latent_dim).to(device)





train_from_model(CAE().to(device))
train_vca(CVAE().to(device))
train_CAAE(encoder, decoder, discriminator)