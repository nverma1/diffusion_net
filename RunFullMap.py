import numpy as np
import scipy.linalg
import keras.backend as K
import tensorflow as tf
import sklearn.manifold
import sklearn.metrics
from scipy.sparse.csgraph import laplacian as csgraph_laplacian

import Diffusion as df
import os.path
from autoencoder import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from keras.utils import np_utils
from keras.layers import Dense, Input
from keras.models import Model
from keras import regularizers


import LaplacianEigenmaps




class DiffusionNet:
    def __init__(self, training_data, embedding_size, k=16, N_EPOCHS=2000, visual=False, embedding='normal'):
        print ("Gpu available: ", tf.test.is_gpu_available())
        S1_train = training_data
        n_train = S1_train.shape[0]
        input_size = S1_train.shape[1]
        batch_size = S1_train.shape[0]                

        if embedding=='laplacian':
            Idx, Dx = df.Knnsearch(S1_train, S1_train, k)
            adj, _ = df.ComputeKernel(Idx, Dx)
            adj = adj - np.eye(n_train)  # this is adjacency, remove 1 because everyone is own neighbor
            E, v = LaplacianEigenmaps.spectral_embedding(adj, n_components=embedding_size, norm_laplacian=False)
            #print (E2.shape, v2.shape)
            new_embedding = np.matmul(E, np.diag(v))
            #new_embedding /= np.expand_dims(np.sum(np.abs(E2),axis=0),axis=0)
            #print(new_embedding.shape)
            embedding = new_embedding

            embedding_matrix = csgraph_laplacian(adj, normed=False,return_diag=False)
            print("Using laplacian embedding")
        else:
            K_mat = df.ComputeLBAffinity(S1_train, k, sig=0.1)  # Laplace-Beltrami affinity: D^-1 * K * D^-1
            P = df.makeRowStoch(K_mat)  # markov matrix
            E, v = df.Diffusion(K_mat, nEigenVals=embedding_size + 1)  # eigenvalues and eigenvectors
            S1_embedding = np.matmul(E, np.diag(v))
            embedding = S1_embedding

            embedding_matrix = P


        print("Done embedding")
        encoder_layer_sizes = [40, 20, embedding_size]
        decoder_layer_sizes = [20, 40, input_size]
        autoencoder1 = Autoencoder(input_size=input_size, hidden_size=encoder_layer_sizes[0],
                                   reg_par=1e-10, output_activation='linear', p=0.05, beta=0.01)
        autoencoder1.compile(optimizer='adam')
        autoencoder1.train(S1_train, batch_size=n_train, n_epochs=N_EPOCHS)
        output1 = autoencoder1.predict(S1_train)
        encoder1_train = autoencoder1.encode(S1_train)  # this is the output of the first hidden layer
        autoencoder2 = Autoencoder(input_size=encoder_layer_sizes[0], hidden_size=encoder_layer_sizes[1],
                                   reg_par=1e-7, p=0.1, beta=0.01)
        autoencoder2.compile(optimizer='adam')
        autoencoder2.train(encoder1_train, batch_size=n_train, n_epochs=N_EPOCHS)
        encoder2_train = autoencoder2.encode(encoder1_train)  # this is the output of the second hidden layer
        encoder3 = pretrain_regression(encoder2_train, embedding, encoder_layer_sizes[1], encoder_layer_sizes[2],
                                       batch_size, reg_par=1e-7, n_epochs=N_EPOCHS)
        encoder3_train = encoder3.predict(encoder2_train)
        print("Done encoder")

        de_autoencoder1 = Autoencoder(input_size=embedding_size, hidden_size=decoder_layer_sizes[0],
                                      reg_par=1e-4, output_activation='linear', p=0.05, beta=0.01)
        de_autoencoder1.compile(optimizer='adam')
        de_autoencoder1.train(embedding, batch_size=n_train, n_epochs=N_EPOCHS)
        de_output1 = de_autoencoder1.predict(embedding)
        de_encoder1_train = de_autoencoder1.encode(embedding)  # this is the output of the first hidden layer
        de_autoencoder2 = Autoencoder(input_size=decoder_layer_sizes[0], hidden_size=decoder_layer_sizes[1],
                                      reg_par=1e-7, p=0.05, beta=0.01)
        de_autoencoder2.compile(optimizer='adam')
        de_autoencoder2.train(de_encoder1_train, batch_size=n_train, n_epochs=N_EPOCHS)
        de_encoder2_train = de_autoencoder2.encode(de_encoder1_train)
        de_encoder3 = pretrain_regression(de_encoder2_train, S1_train, decoder_layer_sizes[1], decoder_layer_sizes[2],
                                          batch_size, reg_par=1e-10, n_epochs=N_EPOCHS)
        de_encoder3_train = de_encoder3.predict(de_encoder2_train)
        print("Done decoder")

        embedding_matrix = tf.cast(tf.constant(embedding_matrix), tf.float32)
        E1 = tf.cast(tf.constant(E), tf.float32)
        v = tf.cast(tf.constant(v), tf.float32)
        init = tf.constant(autoencoder1.get_weights()[0])
        E_W1 = tf.Variable(init)
        init = tf.constant(autoencoder1.get_weights()[1])
        E_b1 = tf.Variable(init)
        init = tf.constant(autoencoder2.get_weights()[0])
        E_W2 = tf.Variable(init)
        init = tf.constant(autoencoder2.get_weights()[1])
        E_b2 = tf.Variable(init)
        init = tf.constant(encoder3.layers[1].get_weights()[0])
        E_W3 = tf.Variable(init)
        # init diffusion net decoder units from pretrained autoencoders
        init = tf.constant(de_autoencoder1.get_weights()[0])
        D_W1 = tf.Variable(init)
        init = tf.constant(de_autoencoder1.get_weights()[1])
        D_b1 = tf.Variable(init)
        init = tf.constant(de_autoencoder2.get_weights()[0])
        D_W2 = tf.Variable(init)
        init = tf.constant(de_autoencoder2.get_weights()[1])
        D_b2 = tf.Variable(init)
        init = tf.constant(de_encoder3.layers[1].get_weights()[0])
        D_W3 = tf.Variable(init)
        theta_E = [E_W1, E_W2, E_W3, E_b1, E_b2]
        theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2]

        def encoder(x):
            h1 = tf.nn.sigmoid(tf.matmul(x, E_W1) + E_b1)
            h2 = tf.nn.sigmoid(tf.matmul(h1, E_W2) + E_b2)
            h3 = tf.matmul(h2, E_W3)
            return h3
        def decoder(z):
            h1 = tf.nn.sigmoid(tf.matmul(z, D_W1) + D_b1)
            h2 = tf.nn.sigmoid(tf.matmul(h1, D_W2) + D_b2)
            h3 = tf.matmul(h2, D_W3)
            return h3

        X = tf.placeholder(tf.float32, shape=[None, input_size])
        Z = encoder(X)
        Y = tf.placeholder(tf.float32, shape=[None, embedding_size])
        R = decoder(Y)
        # the parameters for training the network
        learning_rate = 1e-2
        n_iters = 12000
        # params for loss function
        reg_par = 1e-4
        eta = 100
        encoder_fidelity_loss = tf.reduce_mean(tf.square(Y - Z))
        encoder_eigen_loss = 0
        for i in range(embedding_size):
            mat = embedding_matrix - v1[i] * np.eye(n_train, dtype=np.float32)
            z_vec = tf.slice(Z, [0, i], [-1, 1])
            vec = tf.matmul(mat, z_vec)
            encoder_eigen_loss += tf.reduce_mean(tf.square(vec))
        encoder_reg = tf.nn.l2_loss(E_W3)  # +tf.nn.l2_loss(E_W1) + tf.nn.l2_loss(E_W2)
        encoder_loss = encoder_fidelity_loss + eta * encoder_eigen_loss + reg_par * encoder_reg
        decoder_reg = tf.nn.l2_loss(D_W3)  # +tf.nn.l2_loss(D_W1) + tf.nn.l2_loss(D_W2)
        decoder_loss = tf.reduce_mean(tf.square(X - R)) + reg_par * decoder_reg
        E_solver = (tf.train.AdamOptimizer(learning_rate=learning_rate)
                    .minimize(encoder_loss, var_list=theta_E))
        D_solver = (tf.train.AdamOptimizer(learning_rate=learning_rate)
                    .minimize(decoder_loss, var_list=theta_D))
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for i in range(n_iters):
            if i % 1500 == 0 and visual:  # plot intermediate outputs
                z = sess.run(Z, feed_dict={X: S1_train})
                loss = np.mean(np.sum(np.abs(embedding - z) ** 2, axis=1) ** (1. / 2))
                print('step %.d, loss %.2e' % (i, loss))
                fig, (a2) = plt.subplots(1, 1)
                a2.scatter(z[:, 0], z[:, 1])
                plt.axis('equal')
                plt.title('diffusion net encoder output, iter= ' + str(i))
                plt.show()
            _ = sess.run(E_solver, feed_dict={X: S1_train, Y: embedding})

        for i in range(n_iters):
            if i % 1500 == 0 and visual:  # plot intermediate outputs
                x = sess.run(R, feed_dict={Y: embedding})
                loss = np.mean(np.sum(np.abs(S1_train - x) ** 2, axis=1) ** (1. / 2))
                print('step %.d, loss %.2e' % (i, loss))
                fig, (a2) = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
                a2.scatter(x[:, 0], x[:, 1], x[:, 2])
                plt.axis('auto')
                plt.title('diffusion net decoder output, iter= ' + str(i))
                plt.show()

            _ = sess.run(D_solver, feed_dict={X: S1_train, Y: embedding})
        self.Z=Z
        self.R=R
        self.X=X
        self.Y=Y
        self.sess=sess
    def predict(self, data):
        network_embedding = self.sess.run(self.Z, feed_dict={self.X: data})
        return self.sess.run(self.R, feed_dict={self.Y: network_embedding})