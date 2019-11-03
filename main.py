import os
import spotipy
import spotipy.util as util
import numpy as np
from json.decoder import JSONDecodeError


username = "bbw462"
user_playlist = "TestDataset"

xout = "x1_test.csv"
yout = "y1_test.csv"
scope = 'playlist-read-private'


def show_tracks(results):
    for i, item in enumerate(results['items']):
        track = item['track']
        print("   %d %32.32s %s" % (i, track['artists'][0]['name'], track['name']))


try:
    token = util.prompt_for_user_token(username,scope,client_id='d7f8b5638bde46cd9f6089a637586e61',client_secret='e04315c74de9416599a087895e24b01b',redirect_uri='http://localhost/')
except (AttributeError, JSONDecodeError):
    os.remove(f".cache-{username}")
    token = util.prompt_for_user_token(username,scope,client_id='d7f8b5638bde46cd9f6089a637586e61',client_secret='e04315c74de9416599a087895e24b01b',redirect_uri='http://localhost/')

song_ids = []
badfeatures = []

if token:
    sp = spotipy.Spotify(auth=token)
    playlists = sp.user_playlists(username)
    for playlist in playlists['items']:
        if playlist['name'] == user_playlist:
            results = sp.user_playlist(username, playlist['id'], fields="tracks,next")
            tracks = results['tracks']
            show_tracks(tracks)
            for item in tracks['items']:
                track = item['track']
                song_ids.append(track['id'])
                a = track['id']
                feat = sp.audio_features(a)[0]
                badfeatures.append(feat)
            while tracks['next']:
                    tracks = sp.next(tracks)
                    show_tracks(tracks)
                    for item in tracks['items']:
                        track = item['track']
                        song_ids.append(track['id'])
                        a = track['id']
                        feat = sp.audio_features(a)[0]
                        badfeatures.append(feat)
else:
    print("Can't get token for", username)

features = []

bady = [[]]
for i in range(len(badfeatures)):
    if i <= len(badfeatures)//4:
        bady[0].append(1)
    else:
        bady[0].append(0)

for i in range(len(badfeatures)):
    if badfeatures[i] is None:
        bady[0][i] = 2
    else:
        features.append(badfeatures[i])


y = [[]]
for i in range(len(badfeatures)):
    if bady[0][i] == 2:
        pass
    else:
        y[0].append(bady[0][i])

y = np.array(y)


n = len(features)
x = np.zeros((8, n))


for i in range(len(features)):
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
print(y)

np.savetxt(xout, x, delimiter=",")
np.savetxt(yout, y, delimiter=",")








