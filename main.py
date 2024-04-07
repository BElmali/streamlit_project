import streamlit as st
import pandas as pd
import joblib
import numpy as np
import requests
import random
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Steam Game Recommend", page_icon=":dark_sunglasses:")

@st.cache(allow_output_mutation=True)
def get_data():
    dataframe = pd.read_csv('data.csv')
    dataframe_games = pd.read_csv('games.csv')
    return dataframe, dataframe_games


@st.cache(allow_output_mutation=True)
def get_pipeline():
    pipeline = joblib.load('knn_game_recommender_pipeline.pkl')
    return pipeline

@st.cache(allow_output_mutation=True)
def fetch_image_as_bytes(steam_id: int):
    url = f'https://cdn.akamai.steamstatic.com/steam/apps/{steam_id}/header.jpg'
    response = requests.get(url)
    return response.content


st.title(":rainbow[Steam Game Recommend]")
main_tab, mainrecom_tab,random_tab, recommendation_tab = st.tabs(["Ana Sayfa","Öneri Sistemi", "Rastgele Oyunlar", "Türlere göre öneri sistemi"])


# Rastgele

df,df_games = get_data()

col1, col2, col3, col4, col5 = random_tab.columns(5, gap="small")
columns = [col1, col2, col3, col4, col5]
empty_col1, empty_col2, empty_col3 = random_tab.columns([4,3,2])

if empty_col2.button("Rastgele Oyun Öner"):
    random_game = df_games.loc[(df_games['positive_ratio']>=85) & (df_games['user_reviews']>1500)]
    random_game = df[~df["title"].isna()].sample(5)

    for i, col in enumerate(columns):

        col.image(fetch_image_as_bytes(random_game.iloc[i]['app_id']))
        col.write(f"**{random_game.iloc[i]['title']}**")
empty_col1.write("Pozitif inceleme oranı en az %85 ve kullanıcı incelemesi 1500'den fazla olan oyunlar baz alınmıştır.")
# Öneri Sistemi

pipeline = get_pipeline()
drop_columns=['2D', '3D',  'Anime',  'Co-op', 'Colorful',  'Comedy',
       'Cute', 'Difficult', 'Early Access',
       'Exploration',  'Family Friendly', 'Fantasy',
       'Female Protagonist', 'First-Person', 'Free to Play', 'Funny',
       'Gore', 'Great Soundtrack', 'Horror',
        'Open World',  'Pixel Graphics',
       'Platformer',
       'Relaxing', 'Retro',
       'Sci-fi', 'Shooter',
       'Third Person',  'Violent',
       'app_id', 'tags', 'title']
recom_df=df.drop(columns=drop_columns, axis=1)
col_features1, col_features2, col_recommendation = recommendation_tab.columns(3)
Indie = col_features1.checkbox("Indie", value=False)
Singleplayer = col_features1.checkbox("Singleplayer", value=False)
Casual = col_features1.checkbox("Casual", value=False)
Action = col_features1.checkbox("Action", value=False)
Simulation = col_features1.checkbox("Simulation", value=False)
Atmospheric = col_features1.checkbox("Atmospheric", value=False)
Strategy = col_features1.checkbox("Strategy", value=False)
RPG = col_features2.checkbox("RPG", value=False)
Story_Rich = col_features2.checkbox("Story Rich", value=False)
Puzzle = col_features2.checkbox("Puzzle", value=False)
Multiplayer = col_features2.checkbox("Multiplayer", value=False)
Arcade = col_features2.checkbox("Arcade", value=False)
Adventure = col_features2.checkbox("Adventure",value=False)

features = np.array([int(Indie), int(Singleplayer), int(Casual), int(Action),
                     int(Simulation), int(Atmospheric), int(Strategy),
                     int(RPG),int(Puzzle), int(Multiplayer),
                     int(Arcade),int(Adventure),
                     int(Story_Rich)]).reshape(1, -1)

if col_features2.button("Öneri Getir!"):
    distances, indices = pipeline.named_steps['knn'].kneighbors(pipeline.named_steps['scaler'].transform(features), n_neighbors=20)
    recommended_index = random.choice(indices[0][1:])
    recommended_game = df.iloc[recommended_index]
    col_recommendation.image(fetch_image_as_bytes(recommended_game['app_id']))
    col_recommendation.write(f"**{recommended_game['title']}**")




#Anasayfa
col1, col2, col3 = mainrecom_tab.columns(3)
with col1:
    games_list = df['title'].tolist()
    selected_games = col1.multiselect("Oyun ismi girin veya seçin:", games_list)
    if col1.button("Benzer Oyunları Bul"):
        from sklearn.metrics.pairwise import cosine_similarity
        for selected_game in selected_games:
            selected_game_index = df[df['title'] == selected_game].index[0]
            similarity_scores = cosine_similarity(recom_df, recom_df.iloc[[selected_game_index]])
            similar_games_indices = np.argsort(similarity_scores.squeeze())[::-1][1:5]
            similar_games = df.iloc[similar_games_indices]
            for index, game in similar_games.iterrows():
                col2.image(fetch_image_as_bytes(game['app_id']))
                col2.write(f"**{game['title']}**")




col1, col2, col3 =main_tab.columns(3,gap="small")


with col1:
    st.title("Steam Oyunları Veri Seti")
    df_filtered = df.drop(columns=["title", "app_id", "tags"])
    df_games

    st.write("## Yayın Tarihine Göre Oyun Sayısı")
    df_games['date_release'] = pd.to_datetime(df_games['date_release'])
    game_counts_by_date = df_games.groupby(df_games['date_release'].dt.year).size()
    st.line_chart(game_counts_by_date)

with col2:
    # Kullanıcı incelemelerine göre gruplandırma ve başlıkları sıralama
    grouped_df = df_games.groupby('user_reviews').apply(lambda x: x.sort_values(by='user_reviews', ascending=False)).tail(10)
    graph_data = grouped_df[['title', 'rating']].sort_index(ascending=False)
    st.write("## En çok incelenen 10 Oyun")
    st.write(graph_data)


    # Fiyatların histogramı
    st.write("## Fiyat Dağılımı")
    plt.xlabel('$')
    plt.ylabel('Oyun Sayısı')
    st.line_chart(df_games['price_final'].value_counts())


with col3:
    col3.image("logo.jpg")
    data = df_games[['positive_ratio', 'user_reviews']]
    avg_user_reviews = data.groupby('positive_ratio')['user_reviews'].mean()


    st.write("## Positive Ratio'ya Göre Ortalama User Reviews")
    plt.plot(avg_user_reviews.index, avg_user_reviews.values, marker='o', color='red')
    plt.xlabel('Positive Ratio')
    plt.ylabel('Ortalama User Reviews')
    st.pyplot()


    st.write("## Platform Dağılımı")
    platform_counts = df_games[['win', 'mac', 'linux']].apply(pd.value_counts).T
    platform_counts.plot(kind='bar')
    plt.xlabel('Platform')
    plt.ylabel('Oyun Sayısı')

    for i in range(len(platform_counts)):
        total_height = 0
        for j in range(len(platform_counts.columns)):
            count = platform_counts.iloc[i, j]
            plt.text(i, total_height + count / 2, str(int(count)), ha='center', va='center', color='black')
            total_height += count

    st.pyplot()


    all_platforms = df_games[(df_games['win'] == True) & (df_games['mac'] == True) & (df_games['linux'] == True)]
    total_count = len(all_platforms)
    st.write(f"Hem Windows Hem Mac Hem Linux Destekleyen Oyun Sayısı:**{total_count}**")
