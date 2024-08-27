import sqlite3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

data_home = 'E:/download/人工智能/0机器学习/第五模块：机器学习算法建模实战/Python实现音乐推荐系统/'


def do_play_count():
    triplet_dataset = pd.read_csv(data_home + "train_triplets.txt", sep="\t", header=None,
                                  names=["user", "song", "play_count"])
    # print(triplet_dataset.head())
    print(triplet_dataset.shape)
    # print(triplet_dataset.info)

    output_dict = {}
    for row in triplet_dataset.itertuples(index=True, name=None):
        user = row[1]
        play_count = row[3]
        if user in output_dict:
            play_count += output_dict[user]
        output_dict.update({user: play_count})

    output_list = [{'user': user, 'play_count': play_count} for user, play_count in output_dict.items()]
    play_count_df = pd.DataFrame(output_list)
    play_count_df = play_count_df.sort_values('play_count', ascending=False)
    play_count_df.to_csv('./user_playcount_df.csv', index=False)

    print(play_count_df.head(10))

    output_dict = {}
    for row in triplet_dataset.itertuples(index=True, name=None):
        song = row[2]
        play_count = row[3]
        if song in output_dict:
            play_count += output_dict[song]
        output_dict.update({song: play_count})

    output_list = [{'song': song, 'play_count': play_count} for song, play_count in output_dict.items()]
    song_count_df = pd.DataFrame(output_list)
    song_count_df = song_count_df.sort_values('play_count', ascending=False)
    song_count_df.to_csv('./song_playcount_df.csv', index=False)
    print(song_count_df.head(10))


# do_play_count()
def filter_data():
    user_play_count_df = pd.read_csv('./user_playcount_df.csv')
    song_play_count_df = pd.read_csv('./song_playcount_df.csv')
    total_play_count = user_play_count_df['play_count'].sum()

    top_100000_user_df = user_play_count_df.head(100000)
    top_30000_song_df = song_play_count_df.head(30000)

    print((float(top_100000_user_df.play_count.sum()) / total_play_count) * 100)
    print((float(top_30000_song_df.play_count.sum()) / total_play_count) * 100)

    top_100000_user_list = list(top_100000_user_df.user)
    top_30000_song_list = list(top_30000_song_df.song)

    triplet_dataset = pd.read_csv(data_home + "train_triplets.txt", sep="\t", header=None,
                                  names=['user', 'song', 'play_count'])

    triplet_dataset_sub = triplet_dataset[triplet_dataset.user.isin(top_100000_user_list)]
    del triplet_dataset

    triplet_dataset_sub_song = triplet_dataset_sub[triplet_dataset_sub.song.isin(top_30000_song_list)]
    del triplet_dataset_sub
    print(triplet_dataset_sub_song.shape)
    print(triplet_dataset_sub_song.head())
    triplet_dataset_sub_song.to_csv('./triplet_dataset_sub_song.csv', index=False)


# filter_data()
def get_song_info():
    song_play_count_df = pd.read_csv('./song_playcount_df.csv')
    top_30000_song_df = song_play_count_df.head(30000)
    top_30000_song_list = list(top_30000_song_df.song)
    conn = sqlite3.connect(data_home + 'track_metadata.db')

    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    print(cur.fetchall())

    song_info_df = pd.read_sql(con=conn, sql='select * from songs')
    print(song_info_df.head())
    song_info_df_sub = song_info_df[song_info_df.song_id.isin(top_30000_song_list)]
    song_info_df_sub.to_csv('./track_metadata_df_sub.csv', index=False)
    print(song_info_df_sub.shape)


# get_song_info()

def build_song_info():
    triplet_dataset_sub_song = pd.read_csv(filepath_or_buffer=data_home + 'triplet_dataset_sub_song.csv',
                                           encoding="ISO-8859-1")
    track_metadata_df_sub = pd.read_csv(filepath_or_buffer=data_home + 'track_metadata_df_sub.csv',
                                        encoding="ISO-8859-1")

    del track_metadata_df_sub['track_id']
    del track_metadata_df_sub['artist_mbid']

    track_metadata_df_sub.drop_duplicates(subset=['song_id'])

    triplet_dataset_sub_song_merged = pd.merge(triplet_dataset_sub_song, track_metadata_df_sub, how='left',
                                               left_on='song',
                                               right_on='song_id')
    triplet_dataset_sub_song_merged.rename(columns={'play_count': 'listen_count'}, inplace=True)

    del (triplet_dataset_sub_song_merged['song_id'])
    del (triplet_dataset_sub_song_merged['artist_id'])
    del (triplet_dataset_sub_song_merged['duration'])
    del (triplet_dataset_sub_song_merged['artist_familiarity'])
    del (triplet_dataset_sub_song_merged['artist_hotttnesss'])
    del (triplet_dataset_sub_song_merged['track_7digitalid'])
    del (triplet_dataset_sub_song_merged['shs_perf'])
    del (triplet_dataset_sub_song_merged['shs_work'])

    print(triplet_dataset_sub_song_merged.head(n=10))
    popular_songs_top_20 = triplet_dataset_sub_song_merged[['title', 'listen_count']]\
        .groupby('title').sum().reset_index()\
        .sort_values('listen_count', ascending=False).head(n=20)

    objects = (list(popular_songs_top_20['title']))
    # 设置位置
    y_pos = np.arange(len(objects))
    # 对应结果值
    performance = list(popular_songs_top_20['listen_count'])
    # 绘图
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects, rotation='vertical')
    plt.ylabel('Item count')
    plt.title('Most popular songs')

    plt.show()

# build_song_info()
