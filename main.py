import os
import spotipy
import spotipy.util as util
from json.decoder import JSONDecodeError


def show_tracks(tracks):
    for i, item in enumerate(tracks['items']):
        track = item['track']
        print("   %d %32.32s %s" % (i, track['artists'][0]['name'], track['name']))


username = input("Enter the username: ")
user_playlist = input("Enter the playlist that you want a song recommeneded for: ")
scope = 'playlist-read-private'
try:
    token = util.prompt_for_user_token(username,scope,client_id='d7f8b5638bde46cd9f6089a637586e61',client_secret='e04315c74de9416599a087895e24b01b',redirect_uri='http://localhost/')
except (AttributeError, JSONDecodeError):
    os.remove(f".cache-{username}")
    token = util.prompt_for_user_token(username,scope,client_id='d7f8b5638bde46cd9f6089a637586e61',client_secret='e04315c74de9416599a087895e24b01b',redirect_uri='http://localhost/')


if token:
    sp = spotipy.Spotify(auth=token)
    playlists = sp.user_playlists(username)
    song_ids = []
    for playlist in playlists['items']:
        if playlist['name'] == user_playlist:
            results = sp.user_playlist(username, playlist['id'], fields="tracks,next")
            tracks = results['tracks']
            show_tracks(tracks)
            while tracks['next']:
                tracks = sp.next(tracks)
                show_tracks(tracks)
        # if playlist['owner']['id'] == username:
        #     print()
        #     print(playlist['name'])
        #     print('  total tracks', playlist['tracks']['total'])
        #     results = sp.user_playlist(username, playlist['id'],
        #         fields="tracks,next")
        #     tracks = results['tracks']
        #     show_tracks(tracks)
        #     while tracks['next']:
        #         tracks = sp.next(tracks)
        #         show_tracks(tracks)
else:
    print("Can't get token for", username)