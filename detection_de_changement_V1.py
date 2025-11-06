
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
''''
# Charger une image
im_avant = Image.open("image_proche1.jpg").convert("L")  # "L" = niveau de gris
im_apres = Image.open("image_proche2.jpg").convert("L")  # "L" = niveau de gris

# Convertir en matrice NumPy
X1 = np.array(im_avant)
X2 = np.array(im_apres)
'''

real_part = np.load("/home/joshua/projet_ima_radar/traiter_données_SAR/proj_IMA201_SAR_CD_data/denoised_real_pile_Domancy_TSX26_TSX_SM_SLC_HH_ang45.npy")
imag_part = np.load("/home/joshua/projet_ima_radar/traiter_données_SAR/proj_IMA201_SAR_CD_data/denoised_imag_pile_Domancy_TSX26_TSX_SM_SLC_HH_ang45.npy")
print(real_part.shape)
plt.imshow(real_part[0,...,0])
plt.show()

intensity_image = real_part**2 + imag_part**2
I_t1 = intensity_image[5, ..., 0]   # intensité à t1
I_t2 = intensity_image[20, ..., 0]   # intensité à t2

I_t1_log = np.log(I_t1 + 1e-12)
I_t2_log = np.log(I_t2 + 1e-12)

# Limiter les valeurs extrêmes
I_t1_log = np.clip(I_t1_log, a_min=None, a_max=6)
I_t2_log = np.clip(I_t2_log, a_min=None, a_max=6)

# Normaliser pour l'affichage
I_t1_norm = 255 * (I_t1_log - I_t1_log.min()) / (I_t1_log.max() - I_t1_log.min())
I_t2_norm = 255 * (I_t2_log - I_t2_log.min()) / (I_t2_log.max() - I_t2_log.min())

X1 = I_t1_norm.astype(np.uint8)
X2 = I_t2_norm.astype(np.uint8)

Image.fromarray((X1).astype(np.uint8)).save("image1.png")

# Calculer la différence absolue entre les deux images
Xd = np.abs(X2 - X1)
H = Xd.shape[0]
W = Xd.shape[1]

# h adapté à une image en 720*1280
h =8  # taille des blocs (h x h)

def decouper_en_blocs(matrice, h):
    H, W = matrice.shape
    blocs = []
    for i in range(0, H, h):
        for j in range(0, W, h):
            bloc = matrice[i:i+h, j:j+h]
            blocs.append(bloc)
    return blocs

blocs = decouper_en_blocs(Xd, h)
M=len(blocs)
xd=[]
for i in range(M):
    xd.append(blocs[i].flatten())

xd_array = np.stack(xd)  # shape: (M, h*h)
# Calcul de la moyenne pour chaque position (i, j) sur tous les blocs
vecteur_moyen = np.mean(xd_array, axis=0)

delta = np.zeros((M, h*h))
covariance_matrice = np.zeros((h*h, h*h))
for p in range (M):
    delta[p]= xd[p] - vecteur_moyen 
    covariance_matrice += np.outer(delta[p], delta[p])

covariance_matrice = covariance_matrice / M


# Calcul des valeurs propres et vecteurs propres
valeur_propres, vecteurs_propres= np.linalg.eigh(covariance_matrice)

# Tri en ordre décroissant des valeurs propres et des vecteurs associés
idx = np.argsort(valeur_propres)[::-1]
valeur_propres= valeur_propres[idx]
vecteurs_propres = vecteurs_propres[:, idx]

S = 2 #nombre de composantes principales

P = vecteurs_propres[:, :S]  # matrice de projection (h*h, S)

# xd_array shape: (M, h*h)
# P shape: (h*h, S)
# v shape: (M, S)
v = xd_array @ P  # projection de chaque bloc sur l'espace des S vecteurs propres
                  # chaque ligne est le feature vector d'un bloc
print(v.shape)  # (M, S)

# Clustering avec K-means
kmeans = KMeans(n_clusters=2, random_state=0).fit(v)
labels = kmeans.labels_  # shape: (M,)

# Reconstruire la carte de changement
carte_changement = np.zeros((H, W))
idx = 0
for i in range(0, H, h):
    for j in range(0, W, h):
        if i+h <= H and j+h <= W:
            carte_changement[i:i+h, j:j+h] = labels[idx]
            idx += 1


# Sauvegarder la carte de changemente


Image.fromarray((carte_changement*255).astype(np.uint8)).save("change_map.png")

Image.fromarray((Xd).astype(np.uint8)).save("difference.png")

