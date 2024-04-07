import streamlit as st
import joblib
import numpy as np
import pandas as pd


st.set_page_config(layout = "wide", page_title="Steam Game Recommend", page_icon="🎷")



@st.cache_data
def get_data():
    metadata = pd.read_json('games_metadata.json', lines=True)
    metadata['tags'] = metadata['tags'].apply(lambda x: ', '.join(x))
    metadata['tags'] = metadata['tags'].apply(lambda x: np.nan if x == '' else x)
    metadata.dropna(inplace=True)
    games = pd.read_csv('games.csv')
    content_recom = pd.merge(games, metadata, on='app_id')
    relevant_cols = content_recom[['app_id', 'title', 'tags']]
    relevant_cols = pd.DataFrame(relevant_cols)
    return relevant_cols

@st.cache_data
def get_cosine_sim():
    cosine_sim = joblib.load("cosine_sim.pkl")
    return cosine_sim



st.title(":rainbow[Steam Game Recommend]")



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



# Kullanıcıdan input alınması
input_from_user = st.text_input("Lütfen öneri almak istediğiniz içeriği girin: ")

# Öneri sisteminin çalıştırılması

if input_from_user:
    recommendations = content_based_recommender(input_from_user, get_cosine_sim(), get_data())
    st.write("Önerilen oyunlar:")
    st.write(recommendations)