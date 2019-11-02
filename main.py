
import spotipy
import spotipy.util as util

# util.prompt_for_user_token(username,scope,client_id='d7f8b5638bde46cd9f6089a637586e61',client_secret='',redirect_uri='your-app-redirect-url')



def show_tracks(tracks):
    for i, item in enumerate(tracks['items']):
        track = item['track']
        print("   %d %32.32s %s" % (i, track['artists'][0]['name'], track['name']))


username = input("Enter the username: ")
user_playlist = input("Enter the playlist that you want a song recommeneded for: ")
scope = ""

token = util.prompt_for_user_token(username,client_id='d7f8b5638bde46cd9f6089a637586e61',client_secret='',redirect_uri='http://localhost/')

if token:
    sp = spotipy.Spotify(auth=token)
    playlists = sp.user_playlists(username)
    for playlist in playlists['items']:
        if playlist['owner']['id'] == username:
            print()
            print(playlist['name'])
            print('  total tracks', playlist['tracks']['total'])
            results = sp.user_playlist(username, playlist['id'],
                fields="tracks,next")
            tracks = results['tracks']
            show_tracks(tracks)
            while tracks['next']:
                tracks = sp.next(tracks)
                show_tracks(tracks)
else:
    print("Can't get token for", username)