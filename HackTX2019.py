#Get data from Spotify

import os
import spotipy
import spotipy.util as util
import numpy as np
from json.decoder import JSONDecodeError


username = "kevinphilipc"
user_playlist = "Train"
scope = 'playlist-read-private'
try:
    token = util.prompt_for_user_token(username,scope,client_id='d7f8b5638bde46cd9f6089a637586e61',client_secret='e04315c74de9416599a087895e24b01b',redirect_uri='http://localhost/')
except (AttributeError, JSONDecodeError):
    #os.remove(f".cache-{username}")
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
X = np.zeros((8, n))

for i in range(n):
    dance = features[i]['danceability']
    energy = features[i]['energy']
    mode = float(features[i]['mode'])
    speech = features[i]['speechiness']
    acoustic = features[i]['acousticness']
    instrumental = features[i]['instrumentalness']
    liveness = features[i]['liveness']
    valence = features[i]['valence']

    X[0][i] = dance
    X[1][i] = energy
    X[2][i] = mode
    X[3][i] = speech
    X[4][i] = acoustic
    X[5][i] = instrumental
    X[6][i] = liveness
    X[7][i] = valence



Y = np.zeros((n, 1))
for i in range(n//2):
    Y[i][0] = 1



username = "kevinphilipc"
user_playlist = "Test"
scope = 'playlist-read-private'
try:
    token = util.prompt_for_user_token(username,scope,client_id='d7f8b5638bde46cd9f6089a637586e61',client_secret='e04315c74de9416599a087895e24b01b',redirect_uri='http://localhost/')
except (AttributeError, JSONDecodeError):
    #os.remove(f".cache-{username}")
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

m = len(song_ids)
X_test = np.zeros((8, m))

for i in range(m):
    dance = features[i]['danceability']
    energy = features[i]['energy']
    mode = float(features[i]['mode'])
    speech = features[i]['speechiness']
    acoustic = features[i]['acousticness']
    instrumental = features[i]['instrumentalness']
    liveness = features[i]['liveness']
    valence = features[i]['valence']

    X_test[0][i] = dance
    X_test[1][i] = energy
    X_test[2][i] = mode
    X_test[3][i] = speech
    X_test[4][i] = acoustic
    X_test[5][i] = instrumental
    X_test[6][i] = liveness
    X_test[7][i] = valence



Y_test = np.zeros((m, 1))
for i in range(m//2):
    Y_test[i][0] = 1

Y= Y.T
Y_test = Y_test.T

#-------------------------------------------------------------------------------------------------------------------------------
# Package imports
import numpy as np


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
learning_rates = [1.2, 0.1, 0.01, 0.001]
for lr in learning_rates:
    # Build model
    print('With learning rate of: ' + str(lr))
    parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True, learning_rate = lr)
    print(" ")
    # Print accuracy
    predictions = predict(parameters, X)
    print ('Training Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')
    predictions = predict(parameters, X_test)
    print ('Test Accuracy: %d' % float((np.dot(Y_test,predictions.T) + np.dot(1-Y_test,1-predictions.T))/float(Y.size)*100) + '%')
    print(" ")





