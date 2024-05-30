import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
from math import sqrt
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
import datetime
from sqlalchemy import text
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold



# Connexion à la base de données
DATABASE_URI = 'mysql+pymysql://root:@localhost/nexthoperecrut'
engine = create_engine(DATABASE_URI)

# Fonction pour calculer le RMSE
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))



# rmse_scorer = make_scorer(rmse_score, greater_is_better=False)
# Chargement des données
query = "SELECT job_offer_id, user_id, view_count FROM offer_view;"
data = pd.read_sql(query, engine)

query_applications = "SELECT candidate_id, job_id FROM application;"
applications_data = pd.read_sql(query_applications, engine)

# Création d'une matrice binaire candidat-offre pour les candidatures
applications_matrix = applications_data.pivot(index='candidate_id', columns='job_id', values='candidate_id').notnull().astype(int)

# Création de la matrice utilisateur-offre
data_matrix = data.pivot(index='user_id', columns='job_offer_id', values='view_count').fillna(0)
user_job_matrix = data_matrix.values

# Assurez-vous que applications_matrix a les mêmes colonnes que data_matrix
applications_matrix = applications_matrix.reindex(columns=data_matrix.columns, fill_value=0)

# Division des données en ensembles d'entraînement et de test
train_data, test_data = train_test_split(user_job_matrix, test_size=0.2, random_state=42)

# Définir un poids pour les candidatures existantes
WEIGHT = 10 
# Fonction pour augmenter les scores basés sur les candidatures existantes
def recommend_with_applications(user_id, nmf_model, data_matrix, applications_matrix, n_recommendations=5):
    if user_id in data_matrix.index:
        user_features = nmf_model.transform(data_matrix.loc[[user_id]].values)
        user_scores = np.dot(user_features, nmf_model.components_)

        # Augmentez les scores pour les offres d'emploi où l'utilisateur a déjà postulé
        if user_id in applications_matrix.index:
            user_applications = applications_matrix.loc[user_id]
            user_scores += user_applications.fillna(0).values * WEIGHT  # Ajoutez un poids pour les offres d'emploi déjà postulées

        sorted_job_indices = np.argsort(-user_scores.flatten())[:n_recommendations]
        recommended_job_ids = data_matrix.columns[sorted_job_indices].tolist()
        return recommended_job_ids
    else:
        return []


# # Test de différentes valeurs pour n_components et enregistrement des RMSE
n_components_values = [35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60]  # Vous pouvez choisir les valeurs que vous souhaitez tester
train_rmses = []
test_rmses = []

# Définir le nombre de folds
n_folds = 5

# Créer l'objet KFold
kf = KFold(n_splits=n_folds)

# Initialiser les listes pour stocker les RMSE de chaque fold
train_rmses = np.zeros((len(n_components_values), n_folds))
test_rmses = np.zeros((len(n_components_values), n_folds))

# # Boucle sur chaque valeur de n_components
for i, n_components in enumerate(n_components_values):
    # Boucle sur chaque fold
    for j, (train_index, test_index) in enumerate(kf.split(user_job_matrix)):
        # Sélectionner les données d'entraînement et de test pour ce fold
        train_data, test_data = user_job_matrix[train_index], user_job_matrix[test_index]

        # Appliquer NMF avec le nombre actuel de composants
        nmf = NMF(n_components=n_components, init='random', random_state=0)
        W_train = nmf.fit_transform(train_data)
        H = nmf.components_

        # Calculer les prédictions pour l'ensemble d'entraînement et de test
        train_predictions = np.dot(W_train, H)
        test_predictions = np.dot(nmf.transform(test_data), H)

        # Calculer le RMSE pour les prédictions actuelles
        train_rmse_value = rmse(train_predictions, train_data)
        test_rmse_value = rmse(test_predictions, test_data)
        train_rmses[i, j] = train_rmse_value
        test_rmses[i, j] = test_rmse_value

    # Afficher les résultats pour la valeur actuelle de n_components
    print(f'n_components: {n_components}')
    print(f'Train RMSE: {np.mean(train_rmses[i, :])}')
    print(f'Test RMSE: {np.mean(test_rmses[i, :])}')

# for n_components in n_components_values:
#     # Application de NMF avec le nombre actuel de composants
#     nmf = NMF(n_components=n_components, init='random', random_state=0)
#     W_train = nmf.fit_transform(train_data)
#     H = nmf.components_

#     # Calcul des prédictions pour l'ensemble d'entraînement et de test
#     train_predictions = np.dot(W_train, H)
#     test_predictions = np.dot(nmf.transform(test_data), H)

#     # Calcul du RMSE pour les prédictions actuelles
#     train_rmse_value = rmse(train_predictions, train_data)
#     test_rmse_value = rmse(test_predictions, test_data)
#     train_rmses.append(train_rmse_value)
#     test_rmses.append(test_rmse_value)
    
#     # Affichage des résultats pour la valeur actuelle de n_components
#     print(f'n_components: {n_components}')
#     print(f'Train RMSE: {train_rmse_value}')
#     print(f'Test RMSE: {test_rmse_value}')

# Tracé de la courbe RMSE
# plt.figure(figsize=(10, 6))
# plt.plot(n_components_values, train_rmses, marker='o', label='Train RMSE')
# plt.plot(n_components_values, test_rmses, marker='s', label='Test RMSE')
# plt.title('Entraîner et tester RMSE pour différents nombres de composants')
# plt.xlabel('Nombre de composants')
# plt.ylabel('RMSE')
# plt.legend()
# plt.show()

# # Fonction pour calculer la reconstruction erreur
# def reconstruction_error(X, W, H):
#     return np.linalg.norm(X - np.dot(W, H), 'fro')

# # Valeurs possibles de n_components
# n_components_values = [70,75,80,85,90,95,100,200,300,400,500,600,700,800,900,1000,1100,1200,1400,1300,1500]
# errors = []

# # Boucle sur les valeurs possibles de n_components
# for n_components in n_components_values:
#     nmf = NMF(n_components=n_components, init='random', random_state=42)
#     W = nmf.fit_transform(train_data)
#     H = nmf.components_
#     error = reconstruction_error(train_data, W, H)
#     errors.append(error)

# # Tracer la reconstruction erreur pour chaque valeur de n_components
# plt.plot(n_components_values, errors, marker='o')
# plt.xlabel('Nombre de composants')
# plt.ylabel('Reconstruction Error')
# plt.title('Reconstruction Error vs. Number of Components')
# plt.show()


# Application de NMF sur l'ensemble d'entraînement
nmf = NMF(n_components=45, init='random', random_state=0)
W_train = nmf.fit_transform(train_data)
H = nmf.components_

# Reconstruction des prédictions sur l'ensemble d'entraînement et de test
train_predictions = np.dot(W_train, H)
W_test = nmf.transform(test_data)
test_predictions = np.dot(W_test, H)

# Calcul du RMSE
print(f'RMSE sur l\'ensemble d\'entraînement: {rmse(train_predictions, train_data)}')
print(f'RMSE sur l\'ensemble de test: {rmse(test_predictions, test_data)}')

# Sélection de tous les utilisateurs uniques de la table offer_view
user_ids_query = "SELECT DISTINCT user_id FROM offer_view;"
user_ids_df = pd.read_sql(user_ids_query, engine)
user_ids = user_ids_df['user_id'].tolist()

def save_recommendations_to_db(user_id, recommended_job_ids):
    with engine.connect() as connection: 
        transaction = connection.begin()  # Début d'une transaction
        try:
            for job_id in recommended_job_ids:
                query = text(f"SELECT * FROM recommandation WHERE postulant_id = :user_id AND recommend_job_id = :job_id")
                existing = connection.execute(query, {'user_id': user_id, 'job_id': job_id}).fetchall()
                
                if not existing:
                    insert_query = text("INSERT INTO recommandation (postulant_id, recommend_job_id) VALUES (:user_id, :job_id)")
                    connection.execute(insert_query, {'user_id': user_id, 'job_id': job_id})
            
            transaction.commit()  # Commit de la transaction
        except Exception as e:
            transaction.rollback()  # Rollback en cas d'erreur
            print(f"Erreur lors de l'enregistrement des recommandations pour l'utilisateur {user_id}: {e}")

def recommend_and_save_for_all_users(nmf_model, data_matrix, applications_matrix, n_recommendations=5):
    all_recommendations = {}
    for user_id in user_ids:
        # Vérifiez si l'utilisateur est présent dans data_matrix pour continuer
        if user_id in data_matrix.index:
            user_features = nmf_model.transform(data_matrix.loc[[user_id]].values)
            user_scores = np.dot(user_features, nmf_model.components_)

            # Vérifiez si l'utilisateur a des candidatures et ajustez les scores en conséquence
            if user_id in applications_matrix.index:
                user_applications = applications_matrix.loc[user_id]
                # Assurez-vous que user_applications a la même forme que user_scores
                user_applications = user_applications.reindex_like(data_matrix.loc[user_id]).fillna(0)
                user_scores += user_applications.values * WEIGHT

            sorted_job_indices = np.argsort(-user_scores.flatten())[:n_recommendations]
            recommended_job_ids = data_matrix.columns[sorted_job_indices].tolist()

            # Enregistrer les recommandations dans la base de données
            save_recommendations_to_db(user_id, recommended_job_ids)

            # Stocker les recommandations dans le dictionnaire pour tous les utilisateurs
            all_recommendations[user_id] = recommended_job_ids
        else:
            # Pour les utilisateurs sans vues, vous pouvez choisir de ne pas générer de recommandations
            # ou générer des recommandations aléatoires / basées sur les offres les plus populaires, etc.
            all_recommendations[user_id] = []

    return all_recommendations


# Génération et enregistrement des recommandations pour tous les utilisateurs, et retourner les recommandations
def recommend_and_save_for_all_users(nmf_model, applications_matrix, data_matrix, n_recommendations=5):
    all_recommendations = {}  # Dictionnaire pour stocker les recommandations pour chaque utilisateur
    for user_id in user_ids:
        if user_id in data_matrix.index:
            recommended_job_ids = recommend_with_applications(
            user_id, nmf_model, data_matrix, applications_matrix, n_recommendations)
            
            # Enregistrer les recommandations dans la base de données
            save_recommendations_to_db(user_id, recommended_job_ids)

            # Stocker les recommandations dans le dictionnaire
            all_recommendations[user_id] = recommended_job_ids

    return all_recommendations  # Retourner le dictionnaire des recommandations

# Assurez-vous que l'utilisateur sélectionné existe dans la matrice des données
user_id = '1499'  # Exemple d'ID utilisateur
if int(user_id) in data_matrix.index:
    # Obtenir les caractéristiques de l'utilisateur et prédire les scores pour toutes les offres
    user_features = nmf.transform(data_matrix.loc[[int(user_id)]].values)
    user_scores = np.dot(user_features, nmf.components_)

    # Tracer l'histogramme des scores prédits
    plt.figure(figsize=(10, 6))
    plt.hist(user_scores.flatten(), bins=30, edgecolor='k', alpha=0.7)
    plt.title(f'Distribution des scores prédits pour l\'utilisateur {user_id}')
    plt.xlabel('Score prédit')
    plt.ylabel('Nombre d\'offres d\'emploi')
    plt.show()
else:
    print(f"L'ID utilisateur {user_id} n'existe pas dans la matrice des données.")



# plt.figure(figsize=(10, 6))
# plt.plot(iterations, rmse_train, label='Entraînement')
# plt.plot(iterations, rmse_test, label='Test')
# plt.xlabel('Itérations')
# plt.ylabel('RMSE')
# plt.title('Evolution du RMSE au fil des itérations')
# plt.legend()
# plt.show()

# Génération et enregistrement des recommandations pour tous les utilisateurs
# Appel de la fonction et récupération des recommandations
# recommendations = recommend_and_save_for_all_users(nmf, data_matrix, applications_matrix, n_recommendations=5)

# # Écriture dans un fichier avec les détails des recommandations
# with open(r'C:\Users\TL\Desktop\I.A\task2.txt', 'a') as file:
#     file.write(f'{datetime.datetime.now()} - Execution du script\n')
#     for user_id, recommended_job_ids in recommendations.items():
#         file.write(f'Utilisateur {user_id}: Recommandations enregistrees - {recommended_job_ids}\n')

