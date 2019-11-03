import os
import spotipy
import spotipy.util as util
import numpy as np
from json.decoder import JSONDecodeError

# Gets 3 playlists and username from the user.
username = input("Enter your Spotify username: ")
liked = input("Enter the name of a Spotify playlist that you like: ")
disliked = input("Enter the name of a Spotify playlist that you don't like: ")
selection = input("Enter the name of a Spotify playlist that you want to explore: ")

# Set manually
scope = 'playlist-read-private'

# Function that is used to add track attributes to features list.
def show_tracks(results):
    for i, item in enumerate(results['items']):
        track = item['track']

# Checks the autentication.
try:
    token = util.prompt_for_user_token(username,scope,client_id='d7f8b5638bde46cd9f6089a637586e61',client_secret='e04315c74de9416599a087895e24b01b',redirect_uri='http://localhost/')
except (AttributeError, JSONDecodeError):
    os.remove(f".cache-{username}")
    token = util.prompt_for_user_token(username,scope,client_id='d7f8b5638bde46cd9f6089a637586e61',client_secret='e04315c74de9416599a087895e24b01b',redirect_uri='http://localhost/')

# Empty id list and possibly faulty dictionary list
liked_song_ids = []
disliked_song_ids = []
selection_song_ids = []

liked_faultfeatures = []
disliked_faultfeatures = []
selection_faultfeatures = []

final_faultnames = []
final_faultartist = []

# Fills liked_faultfeatures with the dictionaries for the songs.
if token:
    sp = spotipy.Spotify(auth=token)
    playlists = sp.user_playlists(username)
    for playlist in playlists['items']:
        if playlist['name'] == liked:
            results = sp.user_playlist(username, playlist['id'], fields="tracks,next")
            tracks = results['tracks']
            show_tracks(tracks)
            for item in tracks['items']:
                track = item['track']
                liked_song_ids.append(track['id'])
                a = track['id']
                feat = sp.audio_features(a)[0]
                liked_faultfeatures.append(feat)
            while tracks['next']:
                    tracks = sp.next(tracks)
                    show_tracks(tracks)
                    for item in tracks['items']:
                        track = item['track']
                        liked_song_ids.append(track['id'])
                        a = track['id']
                        feat = sp.audio_features(a)[0]
                        liked_faultfeatures.append(feat)
else:
    print("Can't get token for", username)

# Fills disliked_faultfeatures with the dictionaries for the songs.
if token:
    sp = spotipy.Spotify(auth=token)
    playlists = sp.user_playlists(username)
    for playlist in playlists['items']:
        if playlist['name'] == disliked:
            results = sp.user_playlist(username, playlist['id'], fields="tracks,next")
            tracks = results['tracks']
            show_tracks(tracks)
            for item in tracks['items']:
                track = item['track']
                disliked_song_ids.append(track['id'])
                a = track['id']
                feat = sp.audio_features(a)[0]
                disliked_faultfeatures.append(feat)
            while tracks['next']:
                    tracks = sp.next(tracks)
                    show_tracks(tracks)
                    for item in tracks['items']:
                        track = item['track']
                        disliked_song_ids.append(track['id'])
                        a = track['id']
                        feat = sp.audio_features(a)[0]
                        disliked_faultfeatures.append(feat)
else:
    print("Can't get token for", username)

# Fills selection_faultfeatures with the dictionaries for the songs.
if token:
    sp = spotipy.Spotify(auth=token)
    playlists = sp.user_playlists(username)
    for playlist in playlists['items']:
        if playlist['name'] == selection:
            results = sp.user_playlist(username, playlist['id'], fields="tracks,next")
            tracks = results['tracks']
            show_tracks(tracks)
            for item in tracks['items']:
                track = item['track']
                selection_song_ids.append(track['id'])
                final_faultnames.append(track['name'])
                final_faultartist.append(track['artists'][0]['name'])
                a = track['id']
                feat = sp.audio_features(a)[0]
                selection_faultfeatures.append(feat)
            while tracks['next']:
                    tracks = sp.next(tracks)
                    show_tracks(tracks)
                    for item in tracks['items']:
                        track = item['track']
                        selection_song_ids.append(track['id'])
                        final_faultnames.append(track['name'])
                        final_faultartist.append(track['artists'][0]['name'])
                        a = track['id']
                        feat = sp.audio_features(a)[0]
                        selection_faultfeatures.append(feat)
else:
    print("Can't get token for", username)

# Empty list of cleaned up features and possibly faulty y values.
liked_features = []
disliked_features = []
selection_features = []

liked_y = [[]]
disliked_y = [[]]

# Fills liked y and cleans up liked features
for i in range(len(liked_faultfeatures)):
    liked_y[0].append(1)

for i in range(len(liked_faultfeatures)):
    if liked_faultfeatures[i] is None:
        liked_y.pop()
    else:
        liked_features.append(liked_faultfeatures[i])

# Fills disliked y and cleans up disliked features
for i in range(len(disliked_faultfeatures)):
    disliked_y[0].append(0)

for i in range(len(disliked_faultfeatures)):
    if disliked_faultfeatures[i] is None:
        disliked_y.pop()
    else:
        disliked_features.append(disliked_faultfeatures[i])

# Cleans up selection features and adds to new list.
final_names = []
final_artists = []
for i in range(len(selection_faultfeatures)):
    if selection_faultfeatures[i] is None:
        pass
    else:
        selection_features.append(selection_faultfeatures[i])
        final_names.append(final_faultnames[i])
        final_artists.append(final_faultartist[i])

# Turns the liked y into arrays.
liked_array = np.array(liked_y)
disliked_array = np.array(disliked_y)

# Creates empty np arrays for all 3.
liked_n = len(liked_features)
disliked_n = len(disliked_features)
selection_n = len(selection_features)

liked_x = np.zeros((8, liked_n))
disliked_x = np.zeros((8, disliked_n))
selection_x = np.zeros((8, selection_n))

# Fills the liked x with its feature values.
for i in range(liked_n):
    dance = liked_features[i]['danceability']
    energy = liked_features[i]['energy']
    mode = float(liked_features[i]['mode'])
    speech = liked_features[i]['speechiness']
    acoustic = liked_features[i]['acousticness']
    instrumental = liked_features[i]['instrumentalness']
    liveness = liked_features[i]['liveness']
    valence = liked_features[i]['valence']

    liked_x[0][i] = dance
    liked_x[1][i] = energy
    liked_x[2][i] = mode
    liked_x[3][i] = speech
    liked_x[4][i] = acoustic
    liked_x[5][i] = instrumental
    liked_x[6][i] = liveness
    liked_x[7][i] = valence

# Fills the disliked x with its feature values.
for i in range(disliked_n):
    dance = disliked_features[i]['danceability']
    energy = disliked_features[i]['energy']
    mode = float(disliked_features[i]['mode'])
    speech = disliked_features[i]['speechiness']
    acoustic = disliked_features[i]['acousticness']
    instrumental = disliked_features[i]['instrumentalness']
    liveness = disliked_features[i]['liveness']
    valence = disliked_features[i]['valence']

    disliked_x[0][i] = dance
    disliked_x[1][i] = energy
    disliked_x[2][i] = mode
    disliked_x[3][i] = speech
    disliked_x[4][i] = acoustic
    disliked_x[5][i] = instrumental
    disliked_x[6][i] = liveness
    disliked_x[7][i] = valence

# Fills the selection x with its feature values.
for i in range(selection_n):
    dance = selection_features[i]['danceability']
    energy = selection_features[i]['energy']
    mode = float(selection_features[i]['mode'])
    speech = selection_features[i]['speechiness']
    acoustic = selection_features[i]['acousticness']
    instrumental = selection_features[i]['instrumentalness']
    liveness = selection_features[i]['liveness']
    valence = selection_features[i]['valence']

    selection_x[0][i] = dance
    selection_x[1][i] = energy
    selection_x[2][i] = mode
    selection_x[3][i] = speech
    selection_x[4][i] = acoustic
    selection_x[5][i] = instrumental
    selection_x[6][i] = liveness
    selection_x[7][i] = valence

# Merges the liked and disliked data sets.
X = np.concatenate((liked_x, disliked_x),axis=1)
Y = np.concatenate((liked_y, disliked_y),axis=1)
X_test = selection_x

#-------------------------------------------------------------------------------
np.random.seed(1) # set a seed so that the results are consistent


def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s


#Shape of data sets
shape_X = X.shape
shape_Y = Y.shape
m = X.shape[1]  # training set size


def layer_sizes(X, Y):
    n_x = len(X[:,0])
    n_h = 4
    n_y = len(Y[:, 0])

    return (n_x, n_h, n_y)



def initialize_parameters(n_x, n_h, n_y):

    W1 = np.random.randn(n_h, n_x)*.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(1, n_h)*.01
    b2 = np.zeros((1,1))


    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (1,1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters



def forward_propagation(X, parameters):

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)


    assert(A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache



def compute_cost(A2, Y, parameters):

    m = Y.shape[1]

    cost = -1/m*np.sum(Y*np.log(A2) + (1-Y)*np.log(1-A2), axis = 1)


    cost = float(np.squeeze(cost))
    assert(isinstance(cost, float))

    return cost



def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]

    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]


    dZ2 = A2 - Y
    dW2 = 1/m*(np.dot(dZ2,A1.T))
    db2 = 1/m*np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = np.dot(W2.T, dZ2)*(1 - np.power(A1, 2))
    dW1 = 1/m*np.dot(dZ1, X.T)
    db1 = 1/m*np.sum(dZ1, axis = 1, keepdims = True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads



def update_parameters(parameters, grads, learning_rate = 1.2):


    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]



    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]



    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False, learning_rate = 1.2):
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]


    parameters = initialize_parameters(n_x, n_h, n_y)




    for i in range(0, num_iterations):


        A2, cache = forward_propagation(X, parameters)


        cost = compute_cost(A2, Y, parameters)


        grads = backward_propagation(parameters, cache, X, Y)


        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters


def predict(parameters, X):

    A2, cache = forward_propagation(X, parameters)
    predictions = A2 > 0.5


    return predictions


#IMPLEMENTATION

# Build model
parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True, learning_rate = 1.2)
print()
# Print accuracy
predictions = predict(parameters, X)
print("Accuracy during training: {} %".format(100 - np.mean(np.abs(predictions - Y)) * 100))
predictions = predict(parameters, X_test)
print()
#-------------------------------------------------------------------------------

# Takes the predictions and matches them to the song and artist names.
final_result = [predictions[0], final_artists, final_names]

# Prints the songs that would be liked.
print("Songs that you would like from " +selection+ " based on your entries.")
print()
for i in range(len(predictions[0])):
    if final_result[0][i] == True:
        print(final_result[1][i] +": "+final_result[2][i])



