import streamlit as st
import pandas as pd
import joblib
import numpy as np


st.set_page_config(layout = "wide", page_title="Steam Game Recommend", page_icon="")

@st.cache_data
def get_data():
    dataframe = pd.read_csv('data.csv')
    return dataframe


@st.cache_data
def get_pipeline():
    pipeline = joblib.load('knn_game_recommender_pipeline.pkl')
    return pipeline




st.title(":rainbow[Steam Game Recommend]")
main_tab, random_tab, recommendation_tab = st.tabs(["Ana Sayfa", "Rastgele Oyunlar", "Öneri Sistemi"])


# Rastgele

df = get_data()

col1, col2, col3, col4, col5 = random_tab.columns(5, gap="small")
columns = [col1, col2, col3, col4, col5]
empty_col1, empty_col2, empty_col3 = random_tab.columns([4,3,2])

if empty_col2.button("Rastgele Oyun Öner"):

    random_songs = df[~df["title"].isna()].sample(5)

    for i, col in enumerate(columns):

        #col.image(random_songs.iloc[i]['url'])
        col.write(f"**{random_songs.iloc[i]['title']}**")

# Öneri Sistemi

pipeline = get_pipeline()
drop_columns=['2D', '3D', 'Action-Adventure', 'Anime',
        'Building', 'Cartoony',
       'Choices Matter', 'Co-op', 'Colorful', 'Combat', 'Comedy',
       'Controller', 'Cute', 'Dark', 'Difficult', 'Early Access',
       'Exploration', 'FPS', 'Family Friendly', 'Fantasy',
       'Female Protagonist', 'First-Person', 'Free to Play', 'Funny',
       'Gore', 'Great Soundtrack', 'Horror',  'Linear',
       'Management', 'Minimalist',  'Multiple Endings',
       'Mystery', 'Nudity', 'Open World', 'Physics', 'Pixel Graphics',
       'Platformer', 'Point & Click', 'Psychological Horror',
       'PvP',  'Realistic', 'Relaxing', 'Retro', 'Sandbox',
       'Sci-fi', 'Shooter',
        'Stylized', 'Survival', 'Tactical',
       'Third Person', 'Top-Down', 'VR', 'Violent', 'Visual Novel',
       'app_id', 'tags', 'title']
recom_df=df.drop(columns=drop_columns, axis=1)
col_features1, col_features2, col_recommendation = recommendation_tab.columns(3)
Indie = col_features1.checkbox("Indie", value=False)
Singleplayer = col_features1.checkbox("Singleplayer", value=False)
Casual = col_features1.checkbox("Casual", value=False)
Action = col_features1.checkbox("Action", value=False)
Simulation = col_features2.checkbox("Simulation", value=False)
Atmospheric = col_features2.checkbox("Atmospheric", value=False)
Strategy = col_features2.checkbox("Strategy", value=False)
RPG = col_features2.checkbox("RPG", value=False)
Story_Rich = col_features2.checkbox("Story Rich", value=False)
Puzzle = col_features2.checkbox("Puzzle", value=False)
Multiplayer = col_features2.checkbox("Multiplayer", value=False)
Sports = col_features2.checkbox("Sports", value=False)
Arcade = col_features2.checkbox("Arcade", value=False)
Adventure = col_features2.checkbox("Adventure",value=False)

features = np.array([int(Indie), int(Singleplayer), int(Casual), int(Action),
                     int(Simulation), int(Atmospheric), int(Strategy),
                     int(RPG),int(Puzzle), int(Multiplayer),int(Sports),
                     int(Arcade),int(Adventure),
                     int(Story_Rich)]).reshape(1, -1)

if col_features2.button("Öneri Getir!"):

    distances, indices = pipeline.named_steps['knn'].kneighbors(pipeline.named_steps['scaler'].transform(features), n_neighbors=6)

    recommended_index = indices[0][1]
    recommended_song = df.iloc[recommended_index]

    #col_recommendation.image(recommended_song['url'])
    col_recommendation.write(f"**{recommended_song['title']}**")


col1, col2 = main_tab.columns(2)
with col1:
    games_list = df['title'].tolist()
    selected_games = col1.multiselect("Oyun ismi girin veya seçin:", games_list)

    if col1.button("Benzer Oyunları Bul"):
        from sklearn.metrics.pairwise import cosine_similarity

        for selected_game in selected_games:
            # Seçilen oyunun index'ini bul
            selected_game_index = df[df['title'] == selected_game].index[0]

            # Oyun ismine göre benzerlik skorlarını hesapla
            similarity_scores = cosine_similarity(recom_df, recom_df.iloc[[selected_game_index]])

            # Benzerlik skorlarını sırala ve en benzer olanları al
            similar_games_indices = np.argsort(similarity_scores.squeeze())[::-1][1:5]

            similar_games = df.iloc[similar_games_indices]

            for index, game in similar_games.iterrows():
                col2.write(f"**{game['title']}**")