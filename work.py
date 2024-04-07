import joblib
import pandas as pd
import numpy as np
import datetime as dt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
"""users = pd.read_csv('users.csv')
recom = pd.read_csv('recommendations.csv')
games.head()
games.shape
users.head()
users.shape
recom.head()
recom.shape"""
#EDA
metadata = pd.read_json('games_metadata.json', lines=True)

metadata['tags'] = metadata['tags'].apply(lambda x: ', '.join(x))
metadata['tags'] = metadata['tags'].apply(lambda x: np.nan if x == '' else x)
metadata.dropna(inplace=True)
metadata.isnull().sum()
#metadata = metadata.set_index('app_id')

games = pd.read_csv('games.csv')
#userlerin hangi oyunlara yorum yaptığı
#recom = pd.read_csv('Miull_proje/recommendations.csv')
#recom.groupby(['app_id', 'user_id']).agg(['sum'])

#recom.groupby('app_id')['user_id'].nunique()

#recom.loc[recom["app_id"]==10]

content_recom = pd.merge(games, metadata, on='app_id')
relevant_cols = content_recom[['app_id',"date_release", 'title', 'positive_ratio', 'user_reviews', 'tags']]
relevant_cols = pd.DataFrame(relevant_cols)
#relevant_cols = relevant_cols.set_index('app_id')

relevant_cols.isnull().sum()
# GameLifetime
#"""relevant_cols['date_release'] = pd.to_datetime(relevant_cols['date_release'])
#today_date = pd.Timestamp(dt.datetime.today().date())
#relevant_cols['Game_lifetime'] = (today_date - relevant_cols['date_release']).dt.days"""
relevant_cols = relevant_cols.drop(columns=['date_release'])
#relevant_cols = relevant_cols.loc[relevant_cols['positive_ratio']>50]
#relevant_cols = relevant_cols.loc[relevant_cols['user_reviews']>=30]
#relevant_cols['tags'] = relevant_cols['tags'].apply(lambda x: ', '.join(x))
#relevant_cols['tags'] = relevant_cols['tags'].apply(lambda x: np.nan if x == '' else x)

relevant_cols.isnull().sum()
#relevant_cols.dropna(inplace=True)
relevant_cols.info()


#contentbased
def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    dataframe['tags'] = dataframe['tags'].fillna('')
    tfidf_matrix = tfidf.fit_transform(dataframe['tags'])
    print(tfidf_matrix.shape)
    print(dataframe['title'].shape)
    cosine_sim = cosine_similarity(tfidf_matrix)
    return cosine_sim


def content_based_recommender(title, cosine_sim, dataframe):
    # index'leri oluşturma
    indices = pd.Series(dataframe.index, index=dataframe['title'])
    indices = indices[~indices.index.duplicated(keep='last')]
    # Girilen oyunun ID'sini alın
    game_id = indices[title]
    # Oyun ID'sine karşılık gelen benzerlik skorlarını alın
    similarity_scores = pd.DataFrame(cosine_sim[game_id], columns=["score"])
    # Kendisi hariç ilk 10 oyunu getirin
    similar_games_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index

    # Önerilen oyunların isimlerini döndürün
    return dataframe.iloc[similar_games_indices]['title']
# Öneri almak için oyun ismini kullanın
cosine_sim = calculate_cosine_sim(relevant_cols)
input_from_user = input("Lütfen öneri almak istediğiniz içeriği girin: ")

recommendations=content_based_recommender(input_from_user, cosine_sim, relevant_cols)
print("Önerilen oyunlar:")
print(recommendations)

import joblib

# Öneri fonksiyonunu joblib ile export edin
joblib.dump(content_based_recommender, "content_based_recommender.pkl")
joblib.dump(cosine_sim, "cosine_sim.pkl")
joblib.dump(relevant_cols, "relevant_cols.pkl")