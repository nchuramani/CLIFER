# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 2018
@author: nc528
"""
import numpy
from keras.layers import Dense, Flatten, BatchNormalization, Lambda, Reshape, LeakyReLU, GaussianDropout,\
    SpatialDropout2D
from keras.models import load_model, Model, Input, Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.utils import np_utils
from KEF.DataLoaders import DataLoader_Dynamic_CAAE_Categorical
from keras.losses import kullback_leibler_divergence
from KEF.CustomObjects import losses
from KEF.Implementations import IModelImplementation
from KEF.CustomObjects import ImageGridGenerator
import keras.backend as K
from keras.initializers import TruncatedNormal, Constant, RandomNormal
from keras.regularizers import l1, l2
import os
import datetime
import tensorflow as tf



class CAAE(IModelImplementation.IModelImplementation):

    @property
    def modelName(self):
        return self._modelName

    @property
    def model(self):
        return self._model

    @property
    def logManager(self):
        return self._logManager

    @property
    def plotManager(self):
        return self._plotManager

    def __init__(self, logManager, modelName, baseDir, plotManager=None, batchSize=64, epochs=100,
                 preProcessingProperties=None):
        self._logManager = logManager
        self._modelName = modelName
        self._plotManager = plotManager
        self._interim_saveDir = baseDir
        self.batchSize = batchSize
        self.numberOfEpochs = epochs
        self.preProcessingProperties = preProcessingProperties
        self.z_shape = self.preProcessingProperties[3]
        self.D_ITERS = 2
        numpy.random.seed(2048)



    def encoder(self, inputShape):

        self.logManager.newLogSession("Implementing Encoder Model: " + str(self.modelName) + "_Encoder")

        """ ******************************************************************************"""
        """ Encoder Model """
        """ ******************************************************************************"""

        encoder_input = Input(shape=inputShape, name="Input") # 96 x 96 x 3

        conv_1 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', strides=2, use_bias=True, activation='relu',
                        kernel_initializer=RandomNormal(stddev=0.025), bias_initializer=Constant(0.0),
                        name="Enc_Conv_1")(encoder_input)  # 48 x 48 x 64

        conv_2 = Conv2D(filters=128, kernel_size=(5, 5), padding='same', strides=2, use_bias=True, activation='relu',
                        kernel_initializer=RandomNormal(stddev=0.025), bias_initializer=Constant(0.0),
                        name="Enc_Conv_2")(conv_1)  # 24 x 24 x 128

        conv_3 = Conv2D(filters=256, kernel_size=(5, 5), padding='same', strides=2, use_bias=True, activation='relu',
                        kernel_initializer=RandomNormal(stddev=0.025), bias_initializer=Constant(0.0),
                        name="Enc_Conv_3")(conv_2)  # 12 x 12 x 256

        conv_4 = Conv2D(filters=512, kernel_size=(5, 5), padding='same',strides=2, use_bias=True, activation='relu',
                        kernel_initializer=RandomNormal(stddev=0.025), bias_initializer=Constant(0.0),
                        name="Enc_Conv_4")(conv_3)  # 6 x 6 x 512

        flatten = Flatten(name="Flatten")(conv_4)  # 6 * 6 * 512

        dense_1 = Dense(1024, activation="relu", use_bias=True, kernel_initializer="glorot_normal",
                        bias_initializer=Constant(0.0), name="Enc_dense_1")(flatten)

        out_latent = Dense(self.z_shape, activation="tanh", name="Enc_dense_latent_size")(dense_1)


        self._model_encoder = Model(inputs=encoder_input, outputs=out_latent, name='encoder')

        self._model_encoder.summary()

        self.logManager.write("--- Plotting and saving the Encoder model at: " + str(self.plotManager.plotsDirectory) +
                              "/" + str(self.modelName) + "_Encoder_plot.png")

        self.plotManager.creatModelPlot(self._model_encoder, str(self.modelName) + "_encoder")

        self.logManager.endLogSession()

    def generator(self):


        self.logManager.newLogSession("Implementing Generator Model: " + str(self.modelName) + "_Generator")

        """ ******************************************************************************"""
        """ Decoder Model """
        """ ******************************************************************************"""


        def concat_label(tensors):

            tensor, label = tensors
            tensor_shape = tensor.get_shape().as_list()
            label_shape = label.get_shape().as_list()
            duplicate = tensor_shape[-1] // label_shape[-1]
            label = label * 2 - 1
            label = tf.tile(label, [1, duplicate])
            return K.concatenate([tensor,label])

        generator_input = Input(shape=[self.z_shape], name="Generator_Input")  # E.g. 100

        condition_emo = Input(shape=(self.cond_shape,), name="Condition_Input")  # Label one-hot

        concat_input = Lambda(function=concat_label)([generator_input, condition_emo])

        dense_1 = Dense(512 * 6 * 6, use_bias=True, bias_initializer=Constant(0.0), activation='relu',
                        kernel_initializer=RandomNormal(stddev=0.025), name="gen_dense_1")(concat_input)  # 1024 * 6 * 6

        reshape = Reshape(target_shape=[6, 6, 512])(dense_1) # 6 x 6 x 1024

        conv_1 = Conv2DTranspose(256, kernel_size=(5, 5), padding='same', strides=2, use_bias=True, activation='relu',
                                 kernel_initializer=RandomNormal(stddev=0.025), bias_initializer=Constant(0.0),
                                 name="G_Conv_1")(reshape) # 12 x 12 x 512

        conv_2 = Conv2DTranspose(128, kernel_size=(5, 5), padding='same', strides=2, use_bias=True, activation='relu',
                                 kernel_initializer=RandomNormal(stddev=0.025), bias_initializer=Constant(0.0),
                                 name="G_Conv_2")(conv_1) # 24 x 24 x 256

        conv_3 = Conv2DTranspose(64, kernel_size=(5, 5), padding='same', strides=2, use_bias=True, activation='relu',
                                 kernel_initializer=RandomNormal(stddev=0.025), bias_initializer=Constant(0.0),
                                 name="G_Conv_3")(conv_2)  # 48 x 48 x 128

        conv_4 = Conv2DTranspose(32, kernel_size=(5, 5), padding='same', strides=2,use_bias=True, activation='relu',
                                 kernel_initializer=RandomNormal(stddev=0.025), bias_initializer=Constant(0.0),
                                 name="G_Conv_4")(conv_3)  # 96 x 96 x 64

        if self.preProcessingProperties[2]:
            units = 1
        else:
            units = 3
        generated_img = Conv2DTranspose(units, kernel_size=(5, 5), padding='same', activation="tanh", use_bias=True,
                                        kernel_initializer=RandomNormal(stddev=0.025), bias_initializer=Constant(0.0),
                                        name="Gen_out")(conv_4)  # 96 x 96 x channels

        self._model_generator = Model(inputs=[generator_input, condition_emo], outputs=generated_img,
                                      name='generator')

        self._model_generator.summary()

        self.logManager.write(
            "--- Plotting and saving the Generator model at: " + str(self.plotManager.plotsDirectory) +
            "/" + str(self.modelName) + "_Generator_plot.png")

        self.plotManager.creatModelPlot(self._model_generator, str(self.modelName) + "_Generator")

        self.logManager.endLogSession()

    def discriminator_Img(self, inputShape):

        self.logManager.newLogSession("Implementing Image Discriminator Model: " + str(self.modelName)
                                      + "_Discriminator")

        """ ******************************************************************************"""
        """ Image Discriminator Model """
        """ ******************************************************************************"""

        def concat_label(tensors):
            tensor, label = tensors
            tensor_shape = tensor.get_shape().as_list()
            label_shape = label.get_shape().as_list()
            duplicate = tensor_shape[-1] // label_shape[-1]
            label = label * 2 - 1
            label = tf.tile(label, [1, duplicate])
            label_shape = label.get_shape().as_list()
            x = K.ones_like(tensor)
            x = x[:,:,:,0:label_shape[-1]]
            label = tf.reshape(label, [-1, 1, 1, label_shape[-1]])
            label = label * x
            return K.concatenate([tensor,label],-1)

        condition_emo = Input(shape=(self.cond_shape,), name="Condition_Input")  # Label. e.g. One-hot encoded

        D_input = Input(shape=inputShape, name="Input") # 96 x 96 x channels

        conv_1 = Conv2D(filters=32, kernel_size=(5, 5), padding='same',strides=2, use_bias=True, #activation='relu',
                        kernel_initializer=TruncatedNormal(stddev=0.025), bias_initializer=Constant(0.0),
                        name="DImg_Conv_1")(D_input)  #  48 x 48 x 16
        batch_norm_1 = BatchNormalization(scale=False, name="batch_norm_1")(conv_1)
        act_1 = LeakyReLU(alpha=0.2)(batch_norm_1)

        concatinate_label = Lambda(function=concat_label)([act_1, condition_emo])

        conv_2 = Conv2D(filters=64, kernel_size=(5, 5), padding='same',strides=2, use_bias=True, #activation='relu',
                        kernel_initializer=TruncatedNormal(stddev=0.025), bias_initializer=Constant(0.0),
                        name="DImg_Conv_2")(concatinate_label)  # 24 x 24 x 32

        batch_norm_2 = BatchNormalization(scale=False, name='batch_norm_2')(conv_2)

        act_2 = LeakyReLU(alpha=0.2)(batch_norm_2)

        concatinate_label_1 = Lambda(function=concat_label)([act_2, condition_emo])

        conv_3 = Conv2D(filters=128, kernel_size=(5, 5), padding='same',strides=2, use_bias=True, #activation='relu',
                        kernel_initializer=TruncatedNormal(stddev=0.025), bias_initializer=Constant(0.0),
                        name="DImg_Conv_3")(concatinate_label_1)  # 12 x 12 x 64

        batch_norm_3 = BatchNormalization(scale=False, name="batch_norm_3")(conv_3)
        act_3 = LeakyReLU(alpha=0.2)(batch_norm_3)

        concatinate_label_2 = Lambda(function=concat_label)([act_3, condition_emo])


        conv_4 = Conv2D(filters=256, kernel_size=(5, 5), padding='same',strides=2, use_bias=True, #activation='relu',
                        kernel_initializer=TruncatedNormal(stddev=0.025), bias_initializer=Constant(0.0),
                        name="DImg_Conv_4")(concatinate_label_2)  # 6 x 6 x 128

        batch_norm_4 = BatchNormalization(scale=False, name="batch_norm_4")(conv_4)
        act_4 = LeakyReLU(alpha=0.2)(batch_norm_4)

        flatten = Flatten(name="Flatten")(act_4)  # 6 * 6 * 128

        dense_1 = LeakyReLU(alpha=0.2)(Dense(units=1024, use_bias=True,bias_initializer=Constant(0.0),
                                             kernel_initializer=RandomNormal(stddev=0.02),
                                             name="D_dense_1")(flatten))  # 1028

        out = Dense(1, activation="sigmoid", use_bias=True,bias_initializer=Constant(0.0),
                    kernel_initializer=RandomNormal(stddev=0.025),name="out")(dense_1)

        self._model_discriminator = Model(inputs=[D_input, condition_emo], outputs=out,
                                          name='discriminator')

        self.logManager.write(self._model_discriminator.summary())

        self.logManager.write(
            "--- Plotting and saving the Discriminator model at: " + str(self.plotManager.plotsDirectory) +
            "/" + str(self.modelName) + "_Discriminator_plot.png")

        self.plotManager.creatModelPlot(self._model_discriminator, str(self.modelName) + "_Discriminator")

        self.logManager.endLogSession()

    def discriminator_z(self):

        self.logManager.newLogSession("Implementing Discriminator Z Model: " + str(self.modelName) + "_Dz")

        """ ******************************************************************************"""
        """ Discriminator_Z Model """
        """ ******************************************************************************"""

        D_z_input = Input(shape=[self.z_shape], name="Dz_Input")

        z_dense_1 = Dense(64, activation='relu', use_bias=True,bias_initializer=Constant(0.0),
                          kernel_initializer=RandomNormal(stddev=0.02), name="z_dense_1")(D_z_input)
        z_batch_norm_1 = BatchNormalization(scale=False, name='z_batch_norm_1')(z_dense_1)

        z_dense_2 = Dense(32, activation='relu', use_bias=True,bias_initializer=Constant(0.0),
                          kernel_initializer=RandomNormal(stddev=0.025), name="z_dense_2")(z_batch_norm_1)
        z_batch_norm_2 = BatchNormalization(scale=False, name='z_batch_norm_2')(z_dense_2)

        z_dense_3 = Dense(16, activation='relu', use_bias=True,bias_initializer=Constant(0.0),
                          kernel_initializer=RandomNormal(stddev=0.025), name="z_dense_3")(z_batch_norm_2)
        z_batch_norm_3 = BatchNormalization(scale=False, name='z_batch_norm_3')(z_dense_3)
        z_out = Dense(1, activation='sigmoid', use_bias=True,bias_initializer=Constant(0.0),
                      kernel_initializer=RandomNormal(stddev=0.025),name="z_out")(z_batch_norm_3)

        self._Dz = Model(inputs=D_z_input, outputs=z_out, name='Dz')

        self._Dz.summary()

        self.logManager.write(
            "--- Plotting and saving the Dz model at: " + str(self.plotManager.plotsDirectory) +
            "/" + str(self.modelName) + "_Dz_plot.png")

        self.plotManager.creatModelPlot(self._Dz, str(self.modelName) + "_Dz")

        self.logManager.endLogSession()


    def classifier(self):

        self.logManager.newLogSession("Implementing Classifier C Model: " + str(self.modelName) + "_C")

        """ ******************************************************************************"""
        """ Classifier C Model """
        """ ******************************************************************************"""

        c_input = Input(shape=[self.z_shape], name="c_Input")

        c_dense_1 = Dense(64, activation='relu', use_bias=True,bias_initializer=Constant(0.0),
                          kernel_initializer=RandomNormal(stddev=0.02), name="c_dense_1")(c_input)
        c_batch_norm_1 = BatchNormalization(scale=False, name='z_batch_norm_1')(c_dense_1)

        c_dense_2 = Dense(32, activation='relu', use_bias=True,bias_initializer=Constant(0.0),
                          kernel_initializer=RandomNormal(stddev=0.025), name="c_dense_2")(c_batch_norm_1)
        c_batch_norm_2 = BatchNormalization(scale=False, name='c_batch_norm_2')(c_dense_2)

        c_dense_3 = Dense(16, activation='relu', use_bias=True,bias_initializer=Constant(0.0),
                          kernel_initializer=RandomNormal(stddev=0.025), name="z_dense_3")(c_batch_norm_2)
        c_out = Dense(self.cond_shape, activation='softmax', use_bias=True,bias_initializer=Constant(0.0),
                      kernel_initializer=RandomNormal(stddev=0.025),name="c_out")(c_dense_3)

        self._C = Model(inputs=c_input, outputs=c_out, name='Classifier')

        self._C.summary()

        self.logManager.write(
            "--- Plotting and saving the C model at: " + str(self.plotManager.plotsDirectory) +
            "/" + str(self.modelName) + "_C_plot.png")

        self.plotManager.creatModelPlot(self._C, str(self.modelName) + "_C")

        self.logManager.endLogSession()

    def combineModels_EDz(self):

        self.logManager.newLogSession("Implementing Combined EDz Model: " + str(self.modelName) + "_EDz")
        self._Dz.trainable = False

        encoded_image = self._model_encoder(self._model_encoder.inputs[0])

        dz_out = self._Dz(encoded_image)

        self.combined_EDz = Model(inputs=self._model_encoder.inputs[0], outputs=dz_out, name='EDz')

        self.combined_EDz.summary()

        self.logManager.write(
            "--- Plotting and saving the combined EDz model at: " + str(self.plotManager.plotsDirectory) +
            "/" + str(self.modelName) + "_EDz.png")

        self.plotManager.creatModelPlot(self.combined_EDz, str(self.modelName) + "_EDz")

        self.logManager.endLogSession()

    def combineModels_EC(self):

        self.logManager.newLogSession("Implementing Combined EC Model: " + str(self.modelName) + "_EC")

        encoded_image = self._model_encoder(self._model_encoder.inputs[0])

        c_out = self._C(encoded_image)

        self.combined_EC = Model(inputs=self._model_encoder.inputs[0], outputs=c_out, name='EC')

        self.combined_EC.summary()

        self.logManager.write(
            "--- Plotting and saving the combined EC model at: " + str(self.plotManager.plotsDirectory) +
            "/" + str(self.modelName) + "_EC.png")

        self.plotManager.creatModelPlot(self.combined_EC, str(self.modelName) + "_EC")

        self.logManager.endLogSession()

    def combineModels_GD(self):

        self.logManager.newLogSession("Implementing Combined GD Model: " + str(self.modelName) + "_GD")

        self._model_discriminator.trainable = False

        generated_image = self._model_generator(self._model_generator.inputs)

        D_img_out = self._model_discriminator([generated_image, self._model_generator.inputs[1]])

        self.combined_DG = Model(inputs=self._model_generator.inputs, outputs=D_img_out)

        self.logManager.write(self.combined_DG.summary())

        self.logManager.write(
            "--- Plotting and saving the combined GD model at: " + str(self.plotManager.plotsDirectory) +
            "/" + str(self.modelName) + "_GD.png")

        self.plotManager.creatModelPlot(self.combined_DG, str(self.modelName) + "_GD")

        self.logManager.endLogSession()

    def combineModels_AE(self):

        self.logManager.newLogSession("Implementing  AutoEncoder")

        encoded_image = self._model_encoder(self._model_encoder.inputs[0])

        decoded_image = self._model_generator([encoded_image, self._model_generator.inputs[1]])

        self.combined_AE = Model(inputs=[self._model_encoder.inputs[0], self._model_generator.inputs[1]],
                                 outputs=decoded_image)

        self.logManager.write(self.combined_AE.summary())
        self.logManager.write(
            "--- Plotting and saving the AutoEncoder model at: " + str(self.plotManager.plotsDirectory) +
            "/" + str(self.modelName) + "_AE.png")

        self.plotManager.creatModelPlot(self.combined_AE, str(self.modelName) + "_AE")

        self.logManager.endLogSession()

    def combineModels_cyclic(self):

        self.logManager.newLogSession("Implementing Cyclic System")

        regen_label = Input(shape=(self.cond_shape,), name="Condition_Input_regen")
        autoencoded_image = self.combined_AE([self.combined_AE.inputs[0],self.combined_AE.inputs[1]])
        reconstructed_image = self.combined_AE([autoencoded_image, regen_label])

        self.combined_cyc = Model(inputs=[self.combined_AE.inputs[0], self.combined_AE.inputs[1],
                                          regen_label],
                                 outputs=reconstructed_image)

        self.logManager.write(self.combined_cyc.summary())
        self.logManager.write(
            "--- Plotting and saving the AutoEncoder model at: " + str(self.plotManager.plotsDirectory) +
            "/" + str(self.modelName) + "_cyc.png")

        self.plotManager.creatModelPlot(self.combined_cyc, str(self.modelName) + "_cyc")

        self.logManager.endLogSession()

    def combinedModelFull(self):

        self.logManager.newLogSession("Implementing Combined Model: " + str(self.modelName) + "_full")

        encoder_input = Input(shape=self.inputshape, name="Input") # 96 x 96 x 3
        original_label = Input(shape=(self.cond_shape,), name="Condition_Input_original")
        target_label = Input(shape=(self.cond_shape,), name="Condition_Input_target")


        encoded_image = self._model_encoder(encoder_input)
        dz_out = self._Dz(encoded_image)
        c_out = self._C(encoded_image)

        translated_image = self._model_generator([encoded_image, target_label])
        D_img_out = self._model_discriminator([translated_image, target_label])

        encoded_translated_image = self._model_encoder(translated_image)

        reconstructed_image = self._model_generator([encoded_translated_image, original_label])

        self.combined = Model(inputs=[encoder_input,target_label, original_label],
                              outputs=[dz_out, c_out, D_img_out, translated_image, reconstructed_image])


    def combineModels(self):

        self.combineModels_EDz()
        self.combineModels_EC()
        self.combineModels_GD()
        self.combineModels_AE()
        self.logManager.newLogSession("Implementing Combined Model: " + str(self.modelName) + "_combined")



    def buildModel(self, inputShape,z_shape, cond_shape):
        self.inputshape = inputShape
        self.z_shape = z_shape
        self.cond_shape = cond_shape
        self.logManager.newLogSession("Implementing Model: " + str(self.modelName))
        self.encoder(inputShape=inputShape)
        self.generator()
        self.discriminator_Img(inputShape)
        self.discriminator_z()
        self.classifier()
        self.combineModels()

        self.logManager.endLogSession()

    def train(self, dataPointsTrain, fromEpoch=0, lr=0.0002):


        self.logManager.newLogSession("Training Model on Batches")

        optimizer = Adam(lr=lr, beta_1=0.5, beta_2=0.999,decay=0.)
        self.optimizerType = "Adam"

        self.logManager.write("--- Training Optimizer: " + str(self.optimizerType))

        self.logManager.write("--- Training Strategy: " + str(optimizer.get_config()))

        self.logManager.write("--- Training Batchsize: " + str(self.batchSize))

        self.logManager.write("--- Training Number of Epochs: " + str(self.numberOfEpochs))

        lambs = [0.01, 1, 0.01, 0.01, 0.001]

        self._Dz.compile(loss="binary_crossentropy",
                        loss_weights=[lambs[2]],
                        optimizer=optimizer)
        #
        self.combined_EDz.compile(loss="binary_crossentropy",
                         loss_weights=[lambs[2]],
                         optimizer=optimizer)

        self.combined_EC.compile(loss="categorical_crossentropy",
                        loss_weights=[lambs[3]],
                        optimizer=optimizer)


        self.combined_DG.compile(loss="binary_crossentropy",
                                  loss_weights=[lambs[0]],
                                  optimizer=optimizer)
        #
        self._model_discriminator.compile(loss="binary_crossentropy",
                                          loss_weights=[lambs[0]],
                                          optimizer=optimizer,
                                          metrics=['binary_accuracy'])
        self.combined_AE.compile(loss=losses.EG_loss,
                                 loss_weights=[lambs[1]],
                                 optimizer=optimizer)

        d_losses = []
        dz_losses = []
        g_losses = []
        e_losses = []
        c_losses = []
        ae_losses = []
        cyc_losses = []
        net_losses = []
        d_losses_per_epoch = []
        dz_losses_per_epoch = []
        g_losses_per_epoch = []
        e_losses_per_epoch = []
        c_losses_per_epoch = []
        ae_losses_per_epoch = []
        cyc_losses_per_epoch = []
        net_losses_per_epoch = []
        my_training_batch_generator = DataLoader_Dynamic_CAAE_Categorical.DataGenerator(dataPointsTrain.dataX,
                                                                                        dataPointsTrain.dataY,
                                                                                        self.batchSize,
                                                                                        self.preProcessingProperties,
                                                                                        dataPointsTrain.faces)

        for epoch in range(fromEpoch, self.numberOfEpochs):

            time = datetime.datetime.now()
            numberOfBatches = dataPointsTrain.dataX.shape[0] // self.batchSize
            plotEvery = int(0.05 * numberOfBatches)

            self.logManager.write("Epoch: " + str(epoch+1) + "/" + str(self.numberOfEpochs))
            self.logManager.write("Number of Datapoints: " + str(int(dataPointsTrain.dataX.shape[0])))
            self.logManager.write("Number of Batches: " + str(numberOfBatches))
            self.logManager.write("Plotting after every " + str(plotEvery) + " batches.")

            lrate = lr - lr * (epoch / self.numberOfEpochs)
            K.set_value(self.combined_EDz.optimizer.lr,lrate)
            K.set_value(self.combined_EC.optimizer.lr, lrate)
            K.set_value(self._model_discriminator.optimizer.lr, lrate)
            K.set_value(self._Dz.optimizer.lr, lrate)
            K.set_value(self.combined_DG.optimizer.lr, lrate)
            K.set_value(self.combined_AE.optimizer.lr, lrate)


            for index in range(numberOfBatches):

                self.logManager.write("Batch:" + str(index+1) + "/" + str(numberOfBatches))

                z = numpy.random.uniform(-1, 1, (self.batchSize, self.z_shape))
                input_images, labels = my_training_batch_generator.__getitem__(index)
                labels_save = numpy.array([(numpy.argmax(i) + 1) / self.cond_shape for i in labels]).reshape((-1, 1))


                encoded_z = self._model_encoder.predict(input_images,batch_size=self.batchSize,verbose=0)
                generated_images = self._model_generator.predict([z,labels], verbose=0)

                """Running for DZ"""
                self._Dz.trainable = True

                true_z_label = numpy.ones((self.batchSize, 1))
                fake_z_label = numpy.zeros((self.batchSize, 1))

                dz_temp_fake = self._Dz.train_on_batch(encoded_z,fake_z_label)
                dz_temp_real = self._Dz.train_on_batch(z,true_z_label)

                dz_temp = numpy.average([dz_temp_fake, dz_temp_real])

                """Running for EDz"""
                self._Dz.trainable = False

                edz_temp = self.combined_EDz.train_on_batch(input_images, true_z_label)

                """Running for DImg"""
                self._model_discriminator.trainable = True

                true_labels = numpy.ones((self.batchSize, 1))
                fake_labels = numpy.zeros((self.batchSize, 1))

                dimg_temp_fake, dimg_acc_fake = self._model_discriminator.train_on_batch([generated_images, labels],
                                                                                         fake_labels)

                dimg_temp_real, dimg_acc_real = self._model_discriminator.train_on_batch([input_images, labels],
                                                                                         true_labels)

                dimg_temp = numpy.average([dimg_temp_fake, dimg_temp_real])
                dimg_acc = numpy.average([dimg_acc_fake, dimg_acc_real])

                """Running for GDImg"""
                self._model_discriminator.trainable = False

                gd_temp = self.combined_DG.train_on_batch([z, labels], true_labels)

                """Running for EC"""
                c_loss = self.combined_EC.train_on_batch(input_images, labels)
                lambda_value = numpy.array(sorted([numpy.exp(-1 * (epoch)/2)]) *len(generated_images))
                c_loss_fake = self.combined_EC.train_on_batch(generated_images, labels, sample_weight=[lambda_value] )
                c_loss = numpy.average(c_loss + c_loss_fake)

                """Running for AE"""
                ae_loss = self.combined_AE.train_on_batch([input_images, labels], input_images)

                net_loss = ae_loss + gd_temp + edz_temp + c_loss + dimg_temp + dz_temp

                ae_losses.append(ae_loss)
                g_losses.append(gd_temp)
                e_losses.append(edz_temp)
                c_losses.append(c_loss)
                d_losses.append(dimg_temp)
                dz_losses.append(dz_temp)
                net_losses.append(net_loss)

                self.logManager.write("Batch %d D_Z_loss : %f " % (index + 1, dz_temp))
                self.logManager.write("Batch %d D_Img_loss : %f D_Img_Acc: %.2f" %
                                      (index + 1, dimg_temp, 100. * dimg_acc))
                self.logManager.write("Batch %d E_loss : %f" % (index + 1, edz_temp))
                self.logManager.write("Batch %d AE_loss : %f" % (index + 1, ae_loss))
                self.logManager.write("Batch %d G_loss : %f" % (index + 1, gd_temp))
                self.logManager.write("Batch %d C_loss : %f" % (index + 1, c_loss))
                self.logManager.write("Batch %d Net_loss : %f" % (index + 1, net_loss))
                self._Dz.trainable = True
                self._model_discriminator.trainable = True

                if index % plotEvery == 0:
                    self.save(self._interim_saveDir + "/Model")
                    path = self._interim_saveDir + "/Outputs/Epoch_" + str(epoch) + "/" + str(index) + "_gen.png"
                    path_real = self._interim_saveDir + "/Outputs/Epoch_" + str(epoch) + "/" + str(index) + "_real.png"
                    path_translated = self._interim_saveDir + "/Outputs/Epoch_" + str(epoch) + "/" + str(
                        index) + "_translated.png"

                    generated_images_regen = self.combined_AE.predict([input_images, labels], verbose=0)
                    samp_label_save = numpy.random.randint(0, self.cond_shape, self.batchSize)
                    samp_label = np_utils.to_categorical(samp_label_save, self.cond_shape)
                    generated_images_translated = self.combined_AE.predict( [input_images, samp_label], verbose=0)


                    if self.preProcessingProperties[2]:
                        cmap = 'gray'
                    else:
                        cmap = None
                    ImageGridGenerator.write_image_grid(filepath=path, imgs=generated_images_regen, labels=labels_save,
                                                        label_dict=dataPointsTrain.labelDictionary,
                                                        isGrayscale=self.preProcessingProperties[2], cmap=cmap,
                                                        length=[self.batchSize//self.cond_shape, self.cond_shape])

                    ImageGridGenerator.write_image_grid(filepath=path_real, imgs=input_images, labels=labels_save,
                                                        label_dict=dataPointsTrain.labelDictionary,
                                                        isGrayscale=self.preProcessingProperties[2], cmap=cmap,
                                                        length=[self.batchSize // self.cond_shape,
                                                                self.cond_shape])

                    ImageGridGenerator.write_image_grid(filepath=path_translated, imgs=generated_images_translated,
                                                        labels=samp_label_save,
                                                        label_dict=dataPointsTrain.labelDictionary,
                                                        isGrayscale=self.preProcessingProperties[2], cmap=cmap,
                                                        length=[self.batchSize // self.cond_shape,
                                                                self.cond_shape])

                self.plotManager.plotGenlossBatch({"D": d_losses, "G": g_losses, "AE": ae_losses,
                                              "Dz": dz_losses, "E": e_losses, "C":c_losses,
                                                   "Net": net_losses})
            d_losses_per_epoch.append(numpy.average(d_losses))
            g_losses_per_epoch.append(numpy.average(g_losses))
            ae_losses_per_epoch.append(numpy.average(ae_losses))
            dz_losses_per_epoch.append(numpy.average(dz_losses))
            e_losses_per_epoch.append(numpy.average(e_losses))
            c_losses_per_epoch.append(numpy.average(c_losses))
            cyc_losses_per_epoch.append(numpy.average(cyc_losses))
            net_losses_per_epoch.append(numpy.average(net_losses))
            self.plotManager.plotGenlossBatch({"D_per_epoch": d_losses_per_epoch, "G_per_epoch": g_losses_per_epoch,
                                               "AE_per_epoch": ae_losses_per_epoch, "Dz_per_epoch": dz_losses_per_epoch,
                                               "E_per_epoch": e_losses_per_epoch, "C_per_epoch": c_losses_per_epoch,
                                               "Net_per_epoch": net_losses_per_epoch}, batch=False)
            self.logManager.write("Time taken for epoch " + str(epoch+1) + " : " +
                                  str((datetime.datetime.now() - time).total_seconds()) + " seconds.")
        self.logManager.endLogSession()


    def train_classifier(self,dataPointsTrain, lr=0.0002):

        self.logManager.newLogSession("Training Model")

        optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.optimizerType = "Adam"

        self.logManager.write("--- Training Optimizer: " + str(self.optimizerType))

        self.logManager.write("--- Training Strategy: " + str(optimizer.get_config()))

        self.logManager.write("--- Training Batchsize: " + str(self.batchSize))

        self.logManager.write("--- Training Number of Epochs: " + str(self.numberOfEpochs))

        self.combined_EC.compile(loss="categorical_crossentropy",
                        optimizer=optimizer)
        self.combined_EC.summary()
        training_batch_generator = DataLoader_Dynamic_CAAE_Categorical.DataGenerator(dataPointsTrain.dataX, dataPointsTrain.dataY,
                                                                    self.batchSize, self.preProcessingProperties,
                                                                    dataPointsTrain.faces, dset="MEFED")

        history_callback = self.combined_EC.fit_generator(generator=training_batch_generator,
                                                             steps_per_epoch=(len(dataPointsTrain.dataX) //
                                                                              self.batchSize),
                                                             epochs=5,
                                                             verbose=1,
                                                             shuffle=True,
                                                             use_multiprocessing=True,
                                                             workers=6,
                                                             max_queue_size=16)

        self.logManager.write(str(history_callback.history))
        self.logManager.endLogSession()

    def evaluate(self, dataPoints):
        self.logManager.newLogSession("Model Evaluation")
        evaluation = self.model.evaluate(dataPoints.dataX, dataPoints.dataX, batch_size=self.batchSize)
        self.logManager.write(str(evaluation))
        self.logManager.endLogSession()


    def load_EC(self, loadFolder, cond_shape):
        self.cond_shape = cond_shape
        self._C = load_model(loadFolder + "/" + self.modelName + "_C.h5",
                             custom_objects={"tf": tf})
        self._model_encoder = load_model(loadFolder + "/" + self.modelName + "_Encoder.h5",
                                         custom_objects={"tf": tf, 'EG_loss': losses.EG_loss})

        self._model_encoder.trainable=False
        self.combineModels_EC()
        self.logManager.write("--- Loaded Models from: " + str(loadFolder))

    def save_C(self, saveFolder):

        self.logManager.write("--- Save Folder:" + saveFolder + "/" + self.modelName)

        self._C.save(saveFolder + "/" + self.modelName + "_C.h5")
    def save(self, saveFolder):

        self.logManager.write("--- Save Folder:" + saveFolder + "/" + self.modelName)

        self._model_discriminator.save(saveFolder + "/" + self.modelName + "_Discriminator.h5")
        self._Dz.save(saveFolder + "/" + self.modelName + "_Dz.h5")
        self._model_generator.save(saveFolder + "/" + self.modelName + "_Generator.h5")
        self._model_encoder.save(saveFolder + "/" + self.modelName + "_Encoder.h5")
        self._C.save(saveFolder + "/" + self.modelName + "_C.h5")



    def load(self, loadFolder, cond_shape, GDM=False):
        self.cond_shape = cond_shape

        self._model_generator = load_model(loadFolder + "/" + self.modelName + "_Generator.h5",
                                           custom_objects={"tf": tf, 'EG_loss': losses.EG_loss})
        self._model_encoder = load_model(loadFolder + "/" + self.modelName + "_Encoder.h5",
                                         custom_objects={"tf": tf, 'EG_loss': losses.EG_loss})
        self._model_discriminator = load_model(loadFolder + "/" + self.modelName + "_Discriminator.h5",
                                               custom_objects={"tf": tf,'EG_loss':losses.EG_loss})
        self._Dz = load_model(loadFolder + "/" + self.modelName + "_Dz.h5",
                              custom_objects={"tf": tf, 'EG_loss': losses.EG_loss})
        self._C = load_model(loadFolder + "/" + self.modelName + "_C.h5",
                              custom_objects={"tf": tf})
        self.combineModels()
        self.logManager.write("--- Loaded Models from: " + str(loadFolder))


    def test_EC(self, dataPoint):
        from sklearn.metrics import accuracy_score

        my_training_batch_generator = DataLoader_Dynamic_CAAE_Categorical.DataGenerator(dataPoint.dataX,
                                                                                        dataPoint.dataY,
                                                                                        self.batchSize,
                                                                                        self.preProcessingProperties,
                                                                                        dataPoint.faces)

        num_of_batches = len(dataPoint.dataX) // self.batchSize
        data = []
        labels_gen = []
        for i in range(num_of_batches):
            self.logManager.write("Encoding Batch " + str(i + 1) + "/" + str(num_of_batches))
            images, labels = my_training_batch_generator.__getitem__(i)
            for l in range(len(images)):
                data.append(images[l])
                labels_gen.append(labels[l])

            data = numpy.array(data).reshape((len(data), 96, 96, 3))
            labels_gen = numpy.array(labels_gen).reshape((len(data),self.cond_shape))
            pred_labels = self.combined_EC.predict(x=data, batch_size=self.batchSize)
            labels = [numpy.argmax(l) for l in labels_gen]
            p_labels = [numpy.argmax(l) for l in pred_labels]
            return  accuracy_score(labels, p_labels)


    def translate_emotions(self, dataPoint, dataset):

        my_training_batch_generator = DataLoader_Dynamic_CAAE_Categorical.DataGenerator(dataPoint.dataX,
                                                                                        dataPoint.dataY,
                                                                                        self.batchSize,
                                                                                        self.preProcessingProperties,
                                                                                        dataPoint.faces)
        images, labels = my_training_batch_generator.__getitem__(1)
        generated_images = []

        for j in range(len(images)//7):
            data = images[j]
            data = numpy.expand_dims(data,0)
            generated_images.append(data)
            if self.cond_shape==6:
                self.cond_shape+=1
            for i in range(0,self.cond_shape):
                if i==1:
                    continue
                generated_images.append(self.combined_AE.predict(
                    [data, np_utils.to_categorical(i, self.cond_shape).reshape((1, self.cond_shape))], verbose=0))


        generated_images = numpy.array(generated_images).reshape((-1,96,96,3))
        path_real = self._interim_saveDir + "/Outputs/Generated/Original-Anger-Fear-Happy-Sad-Surprise-Neutral.png"
        directory = os.path.dirname(os.path.abspath(path_real))
        if not os.path.exists(directory):
            os.makedirs(directory)

        ImageGridGenerator.write_image_grid(filepath=path_real, imgs=generated_images,
                                            label_dict=dataPoint.labelDictionary,
                                            isGrayscale=self.preProcessingProperties[2],dset="NONE", length=[self.batchSize//7,7])
    def generate_data(self, dataPoint):

        my_training_batch_generator = DataLoader_Dynamic_CAAE_Categorical.DataGenerator(dataPoint.dataX,
                                                                                        dataPoint.dataY,
                                                                                        self.batchSize,
                                                                                        self.preProcessingProperties,
                                                                                        dataPoint.faces)

        num_of_batches = len(dataPoint.dataX) // self.batchSize
        # remaining = len(dataPoint.dataX) % self.batchSize
        encoded_data = []
        labels_gen = []
        for i in range(num_of_batches):
            self.logManager.write("Encoding Batch " + str(i + 1) + "/" + str(num_of_batches))
            images, labels = my_training_batch_generator.__getitem__(i)
            encoded_data.append(self._model_encoder.predict(images).flatten())
            labels_gen.append([numpy.argmax(l) for l in labels])
        # if remaining > 0:
        #     self.logManager.write("Encoding Remaining ")
        #     images, labels = my_training_batch_generator.remaining(num_of_batches,remaining)
        #     encoded_data.append(self._model_encoder.predict(images).flatten())
        #     labels_gen.append([numpy.argmax(l) for l in labels])
        #     num_of_batches += 1
        encoded_data = numpy.array(encoded_data).reshape(num_of_batches*self.batchSize, self.z_shape)
        labels_gen = numpy.array(labels_gen).reshape(-1, 1)
        return encoded_data, labels_gen.flatten()


    def plotImages(self, dataPoint):
        generated_images = numpy.array(dataPoint).reshape((-1, 96, 96, 3))
        #'Neutral', 'Happy', 'Surprise', 'Disgust', 'Angry', 'Sad', 'Fear'
        path_real = self._interim_saveDir + "/Outputs/Generated/generated.png"
        directory = os.path.dirname(os.path.abspath(path_real))
        if not os.path.exists(directory):
            os.makedirs(directory)

        ImageGridGenerator.write_image_grid(filepath=path_real, imgs=generated_images,
                                            isGrayscale=self.preProcessingProperties[2], dset="NONE",
                                            length=[1, 7])

    def imagine(self, data, elabels, order_label_dict, times, dataset="RAV"):
        encoded_data = []
        label_data = []
        for t in range(times):
            for d in range(len(data)):
                count = 0
                dat = numpy.expand_dims(data[d], 0)
                if dataset.startswith("BAUM"):
                    CAAE_label_dict = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
                else:
                    CAAE_label_dict = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

                for i in range(0, len(CAAE_label_dict)):
                    if i == 1:
                        continue
                    generated_image = self._model_generator.predict([dat,
                                         (np_utils.to_categorical(
                                             i, len(CAAE_label_dict)).reshape((-1, len(CAAE_label_dict))))],
                                                verbose=0)
                    if count==0:
                        encoded_data.append(dat)
                        label_data.append(elabels[0][d])
                    encoded_data.append(self._model_encoder.predict(generated_image))
                    label = [num for num, emo in order_label_dict.items()  if emo == CAAE_label_dict[i]][0]
                    label_data.append(label)
                    if count==0:
                        count=1

        encoded_data = numpy.array(encoded_data).reshape(-1, self.z_shape)
        return encoded_data, numpy.array(label_data).reshape(1,len(label_data))

    def mlp_classify(self, data, elabels, order_label_dict, dataset="RAV"):
        encoded_data = []
        label_data = []
        for d in range(len(data)):
            count = 0
            dat = numpy.expand_dims(data[d], 0)
            if dataset.startswith("BAUM"):
                CAAE_label_dict = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
            else:
                CAAE_label_dict = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

            for i in range(0, len(CAAE_label_dict)):
                if i == 1:
                    continue
                generated_image = self._model_generator.predict([dat,
                                     (np_utils.to_categorical(
                                         i, len(CAAE_label_dict)).reshape((-1, len(CAAE_label_dict))))],
                                            verbose=0)
                if count==0:
                    encoded_data.append(dat)
                    label_data.append(elabels[0][d])
                encoded_data.append(self._model_encoder.predict(generated_image))
                label = [num for num, emo in order_label_dict.items()  if emo == CAAE_label_dict[i]][0]
                label_data.append(label)
                if count==0:
                    count=1

        encoded_data = numpy.array(encoded_data).reshape(-1, self.z_shape)
        return encoded_data, numpy.array(label_data).reshape(1,len(label_data))
