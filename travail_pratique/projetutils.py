""" 
Utilitaires fournis pour faciliter la réalisation du projet/TP du cours d'introduction au réseau de neurones 
Master 2 ISN/FC du Département de Mathématiques de l'Université de Lille (Automne 2018). 
Site web du cours: http://chercheurs.lille.inria.fr/pgermain/neurones2018/index.html
Auteur: Pascal Germain
"""

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import make_grid
import zipfile
import imageio
from itertools import chain
from copy import deepcopy


def charger_cifar(repertoire, etiquettes=None, max_par_etiquettes=None):
    """Charger l'ensemble de données CIFAR

    Paramètres
    ----------
    repertoire: Le répertoire où se trouvent les données
    etiquettes: Une liste contenant des nombres entiers de 0 à 9, précisant les classes à charger
                Par défaut etiquettes=None, ce qui est équivalent à etiquettes=[0,1,2,3,4,5,6,7,8,9]
    max_par_etiquettes: Maximum de données à charger par classe. Par défaut, max_par_etiquettes=None
                        et toutes les données sont chargées.
                        
    Retour
    ------
    Un couple x, y:
        x est une matrice numpy, dont chaque ligne correspond à une image concaténée en un vecteur 
        de 3*32*32=3072 dimensions
            - Les dimensions 0 à 1023 contiennent les valeurs du canal rouge
            - Les dimensions 1024 à 2047 contiennent les valeurs du canal vert
            - Les dimensions 2048 à 3071 contiennent les valeurs du canal bleu
        y est un vecteur contenant la classe de chaque image, soit un entier de 0 à len(etiquettes)-1    
    """
    if etiquettes is None:
         etiquettes = np.arange(10)
    images_list = [None] * len(etiquettes)
    labels_list = [None] * len(etiquettes)
    for i, val in enumerate(etiquettes):
        nom_fichier = repertoire + f'label{val}.zip'
        fichier_zip = zipfile.ZipFile(nom_fichier, "r")
        liste_png = [a for a in fichier_zip.namelist() if a[-4:]=='.png']
        if max_par_etiquettes is not None and len(liste_png) > max_par_etiquettes:
            liste_png = liste_png[:max_par_etiquettes]
         
        nb = len(liste_png)
        data = np.zeros((nb, 32*32*3), dtype=np.float32)
        for j, nom_image in enumerate(liste_png):
            buffer = fichier_zip.read(nom_image)
            image = imageio.imread(buffer)
            r, g, b = [image[:, :, c].reshape(-1) for c in (0,1,2)]
            data[j] = np.concatenate((r,g,b))
        
        images_list[i] = data / 255
        labels_list[i] = i*np.ones(nb, dtype=np.int64)
        print(val, ':', nb, 'images')
        
    x = np.vstack(images_list)
    y = np.concatenate(labels_list)
    print('Total :', len(y), 'images')
    return x, y


def afficher_grille_cifar(images):
    """Affiche une grille d'images provenant de l'ensemble CIFAR (en format numpy.array ou torch.Tensor). 
    Chaque image doit contenir 3*32*32=3072 pixels."""
    plt.figure(figsize=(15,6))
    images3d = torch.Tensor(images).view(-1,3,32,32)
    grid = make_grid(images3d)
    grid = grid.detach().numpy().transpose((1,2,0))
    plt.imshow(grid) 
    
    
class ReseauClassifGenerique:
    """ Classe destinée en encapsuler une architecture de réseau de neurones pour la classification 
    multi-classes. L'apprentissage est effectué par descente de gradient stochastique avec «minibatch»,
    et il est possible de déterminer l'arrêt de l'optimisation par «early stopping».

    Paramètres
    ----------
    architecture: Objet contenant l'architecture du réseau de neurones à optimiser. Doit contenir les
                  méthodes propagation(self, x, apprentissage) et parametres(self).
                  Voir l'exemple de la classe "UneArchiPleinementConnectee"
    eta, alpha: Parametres de la descente en gradient stochastique (taille du gradient et momentum)
    nb_epoques: Nombre d'époques maximum de la descente en gradient stochastique 
    taille_batch: Nombre d'exemples pour chaque «minibatch»
    fraction_validation: Fraction (entre 0.0 à 1.0) des exemples d'apprentissage à utiliser pour
                         créer un ensemble de validation pour le «early stopping».
                         Par défaut fraction_validation=None et il n'y a pas de «early stopping».
    patience: Paramètre de patience pour le «early stopping».
    """
    def __init__(self, architecture, eta=0.4, alpha=0.1, nb_epoques=10, taille_batch=32, 
                 fraction_validation=None, patience=10):
        # Initialisation des paramètres
        self.architecture = architecture
        self.eta = eta
        self.alpha = alpha
        self.nb_epoques = nb_epoques
        self.taille_batch = taille_batch
        self.fraction_validation = fraction_validation
        self.patience = patience
        
        # Ces deux listes serviront à maintenir des statistiques lors de l'optimisation
        self.liste_objectif = list()
        self.liste_validation = list()
        
    def fit(self, x, y):
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int64)              
        nb_sorties = len(torch.unique(y))
                
        if self.fraction_validation is None:
            # Aucun «early stopping»
            early_stopping = False
            
            # Toutes les données sont dédiées à l'apprentissage
            train_data = TensorDataset(x, y)
            max_epoques = self.nb_epoques
        else:
            early_stopping = True
            
            # Création de l'ensemble de validation pour le «early stopping»
            nb_valid = int(self.fraction_validation * len(y))
            nb_train = len(y) - nb_valid
            train_data = TensorDataset(x[:nb_train], y[:nb_train])
            valid_data = TensorDataset(x[nb_train:], y[nb_train:])
            
            # Initialisation des variables pour le «early stopping»
            meilleure_precision = 0.
            meilleure_epoque = 0
            meilleure_archi = deepcopy(self.architecture)
            max_epoques = self.patience
            
        # Initialisation du problème d'optimisation
        sampler = DataLoader(train_data, batch_size=self.taille_batch, shuffle=False) 
        perte_logistique = nn.NLLLoss()       
        optimizer = torch.optim.SGD(chain(*self.architecture.parametres()), 
                                    lr=self.eta, momentum=self.alpha)
           
        # Descente de gradient
        t = 0
        while t < min(max_epoques, self.nb_epoques):
            t += 1
            
            # Une époque correspond à un passage sur toutes les «mini-batch»
            liste_pertes = list()
            for batch_x, batch_y in sampler:
                
                # Propagation avant
                y_pred = self.architecture.propagation(batch_x, apprentissage=True)
                perte = perte_logistique(y_pred, batch_y)

                # Rétropropagation
                optimizer.zero_grad()
                perte.backward()
                optimizer.step()
                
                liste_pertes.append(perte.item())
                
            # Pour fin de consultation future, on conserve les statistiques sur la fonction objectif
            perte_moyenne = np.mean(liste_pertes)
            self.liste_objectif.append(perte_moyenne)
            message = f'[{t}] {perte_moyenne:.5f}'
            
            if early_stopping:
                # Calcule la précision sur l'ensemble de validation
                pred_valid = self.architecture.propagation(valid_data.tensors[0], apprentissage=False)
                pred_valid = torch.argmax(pred_valid, dim=1)
                precision_valid = torch.sum(pred_valid == valid_data.tensors[1]).item() / nb_valid
                self.liste_validation.append(precision_valid)
                message += f' | validation: {precision_valid:3f}'
               
                if precision_valid > meilleure_precision:
                    # Conserve le meilleur modèle 
                    meilleure_precision = precision_valid
                    meilleure_archi = deepcopy(self.architecture)
                    meilleure_epoque = t
                    max_epoques = t + self.patience
                    message += f' ---> meilleur modèle à ce jour (max_t={max_epoques})' 
            
            # Fin de l'époque: affiche le message d'état à l'utilisateur avant de passer à l'époque t+1                     
            print(message)    
        
        print('=== Optimisation terminée ===')
        
        # Dans le cas du «early stopping», on retourne à l'état du modèle offrant la meilleure précision en validation  
        if early_stopping:
            self.architecture = meilleure_archi
            print(f"Early stopping à l'époque #{meilleure_epoque}, avec précision en validation de {meilleure_precision}")
                
    def predict(self, x):
        # On s'assure que les données sont dans le bon format pytorch
        x = torch.tensor(x, dtype=torch.float32)
        
        # Propagation avant 
        pred = self.architecture.propagation(x, apprentissage=False)
        
        # La prédiction correspond à l'indice du neurone de sortie ayant la valeure maximale
        pred = torch.argmax(pred, dim=1)
        
        # On convertit le vecteur de résultat en format numpy
        return np.array(pred.detach())

    
class UneArchiPleinementConnectee:
    """Permets de définir une architecture de réseau pleinement connectée à une couche cachée.
    
    Paramètres:
    -----------
    nb_entrees: La dimension de la couche d'entrée.
    nb_sorties: La dimension de la couche de sortie.
    nb_neurones_cachees: Le nombre de neurones sur la couche cachée du réseau.
    """
    def __init__(self, nb_entrees, nb_sorties, nb_neurones_cachees=50):
        self.modele_plein = nn.Sequential(
            nn.Linear(nb_entrees, nb_neurones_cachees),
            nn.ReLU(),
            nn.Linear(nb_neurones_cachees, nb_sorties),
            nn.LogSoftmax(dim=1)
        )

    def propagation(self, x, apprentissage=False):
        """Effectue la propagation avant des données x dans le réseau."""
        
        # Ce code if/else est superflu pour cet exemple, mais sera essentiel 
        # pour un réseau avec «dropout» ou «batchnorm»
        if apprentissage: 
            self.modele_plein.train()
        else:
            self.modele_plein.eval()
          
        # Propageons la «batch». Notez que nous devons redimensionner nos données consciencieusement
        x1 = self.modele_plein(x)
        return x1
    
    def parametres(self):
        """Retourner une liste contenant toutes les variables à optimiser."""
        return [self.modele_plein.parameters()]
    
    
def compter_parametres(parametres):
    """Calcule le nombre de paramètres à optimiser dans l'architecture d'un réseau"""
    somme = 0
    for p in chain(*parametres):
        nb = 1
        for dimension in p.shape:
            nb *= dimension
        somme += nb
        
    return somme


  