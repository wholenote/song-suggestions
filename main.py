import os
import spotipy
import spotipy.util as util
import numpy as np
from json.decoder import JSONDecodeError

username = "kevinphilipc"
user_playlist = "Favorites"
scope = 'playlist-read-private'
try:
    token = util.prompt_for_user_token(username,scope,client_id='d7f8b5638bde46cd9f6089a637586e61',client_secret='e04315c74de9416599a087895e24b01b',redirect_uri='http://localhost/')
except (AttributeError, JSONDecodeError):
    os.remove(f".cache-{username}")
    token = util.prompt_for_user_token(username,scope,client_id='d7f8b5638bde46cd9f6089a637586e61',client_secret='e04315c74de9416599a087895e24b01b',redirect_uri='http://localhost/')

song_ids = []
features = []

if token:
    sp = spotipy.Spotify(auth=token)
    playlists = sp.user_playlists(username)
    for playlist in playlists['items']:
        if playlist['name'] == user_playlist:
            results = sp.user_playlist(username, playlist['id'], fields="tracks,next")
            tracks = results['tracks']
            for item in tracks['items']:
                track = item['track']
                song_ids.append(track['id'])
                a = track['id']
                feat = sp.audio_features(a)[0]
                features.append(feat)
else:
    print("Can't get token for", username)


#print(song_ids)
#print(features)

n = len(song_ids)
x = np.zeros((8, n))

for i in range(n):
    dance = features[i]['danceability']
    energy = features[i]['energy']
    mode = float(features[i]['mode'])
    speech = features[i]['speechiness']
    acoustic = features[i]['acousticness']
    instrumental = features[i]['instrumentalness']
    liveness = features[i]['liveness']
    valence = features[i]['valence']

    x[0][i] = dance
    x[1][i] = energy
    x[2][i] = mode
    x[3][i] = speech
    x[4][i] = acoustic
    x[5][i] = instrumental
    x[6][i] = liveness
    x[7][i] = valence

print(x)

y = np.zeros((1, n))









