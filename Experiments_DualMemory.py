from KEF.Controllers import ExperimentManager
from KEF.DataLoaders import DataLoader_CAAE_Categorical
from KEF.Implementations.ExternalImplementations.GDM_Imagine.episodic_gwr import EpisodicGWR
from KEF.Implementations.ExternalImplementations.GDM_Imagine import gtls
from keras import backend as K
import tensorflow as tf
import argparse
import numpy as np
import itertools
import pandas as pd
from scipy import mean
from scipy.stats import t, sem, kruskal, anderson_ksamp
import seaborn as sns
import csv
import os
sns.set_style("whitegrid")
import random
import matplotlib.pyplot as plt
import re
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def sort_nicely(l):
    """ Sort the given list in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    l.sort(key=alphanum_key)
    return l

def loadData(dataLoader, datasetFolder, labelCSV, dataSet, dataType="Train", emo=None, subjectID=None, order=None):
    if dataType.startswith("Train"):
        dataLoader.loadTrainData(datasetFolder, labelCSV, dataset=dataSet, personID=subjectID, emo=emo, order=order)
    elif dataType.startswith("Valid"):
        dataLoader.loadValidationData(datasetFolder,labelCSV, dataset=dataSet, personID=subjectID,emo=emo, order=order)
    elif dataType.startswith("Test"):
        dataLoader.loadTestData(datasetFolder, labelCSV,dataset=dataSet, personID=subjectID,emo=emo, order=order)
    return dataLoader


def loadModel(experimentManager, args, loadPath, preProcessingProperties):
    model = CAAE_Imagination_Categorical_Confer.CAAE(experimentManager.logManager, "CAAE_Categorical_",
                               experimentManager.baseDirectory + "/" + experimentManager.experimentName,
                               experimentManager.plotManager, args.batch_size,
                               args.epochs,
                               preProcessingProperties)
    model.load(loadPath, cond_shape=args.classes + 1)
    return model

def train_GDM(data, labels, CAAE_model, trained_GWRs, labelsize, train_replay, train_imagine, order_label_dictionary,
              args=None, mlp_classifier=None, first_classify=False):

    def replay_samples(net, size):
        samples = np.zeros(size, dtype=int)
        r_weights = np.zeros((net.num_nodes, size, net.dimension))
        r_labels = np.zeros((net.num_nodes, len(net.num_labels), size))
        for i in range(0, net.num_nodes):
            for r in range(0, size):
                if r == 0:
                    samples[r] = i
                else:
                    samples[r] = np.argmax(net.temporal[int(samples[r - 1]), :])
                r_weights[i, r] = net.weights[int(samples[r])][0]
                for l in range(0, len(net.num_labels)):
                    r_labels[i, l, r] = np.argmax(net.alabels[l][int(samples[r])])
        return r_weights, r_labels

    experimentManager.logManager.write("Replay: " + str(train_replay))
    experimentManager.logManager.write("Imagination: " + str(train_imagine))
    experimentManager.logManager.write("MLP_Classifier: " + str(mlp_classifier))

    '''
    Episodic-GWR supports multi-class neurons.
    Set the number of label classes per neuron and possible labels per class
    e.g. e_labels = [50, 10]
    is two labels per neuron, one with 50 and the other with 10 classes.
    Setting the n. of classes is done for experimental control but it is not
    necessary for associative GWR learning.
    '''
    # Setting up format for Episodic and Semantic Memory labelling
    e_labels = [labelsize, labelsize]
    s_labels = [labelsize]

    # Training Data
    ds_vectors = data


    # Training Labels
    dataY = labels
    ds_labels = np.zeros((len(e_labels), len(dataY)))
    ds_labels[0] = dataY
    ds_labels[1] = dataY

    # Number of context descriptors; Set to zero for frame-based evaluations.
    num_context = 0

    # Hyper-parameters
    a_threshold = [0.4, 0.2]
    h_thresholds = [0.5, 0.2]
    beta = 0.7
    e_learning_rates = [0.2, 0.001]
    s_learning_rates = [0.02, 0.001]
    context = True

    if mlp_classifier:
        classifier_model, mlp_accs, f1s = test_mlp(args=args, data=data, labels=labels, CAAE_model=CAAE_model,
                                 order=order_label_dictionary, first=first_classify, classifier_model=trained_GWRs[0],
                                                   train=True)
        return (classifier_model, classifier_model), mlp_accs, f1s
    else:
        # Initialising Episodic and Semantic Memory GWR Models.
        if trained_GWRs[0] is None:
            # Initialise Episodic Memory
            g_episodic = EpisodicGWR()
            # Higher Max-nodes and lower age allow for faster learning with pattern-separated representations.
            g_episodic.init_network(data, e_labels, num_context, max_nodes=len(data), age=1000)

            # Initialising Semantic Memory
            g_semantic = EpisodicGWR()
            # Lower Max-nodes and higher age allow for slower learning with pattern-complete representations.
            g_semantic.init_network(data, s_labels, num_context, max_nodes=len(data)//2, age=2000)
        else:
            # Loading trained models for subsequent training runs.
            g_episodic, g_semantic = trained_GWRs

        """ Incremental training hyper-parameters """

        # Epochs per sample for incremental learning
        epochs = 10
        # Initialising experienced episodes to Zero.
        n_episodes = 0
        # Number of samples per epoch
        batch_size = 5

        # Replay parameters; With num_context = 0, RNATs size set to 1, that is, only looking at previous BMU.
        # Size of RNATs
        replay_size = (num_context * 2) + 1
        replay_weights = []
        replay_labels = []

        """##############################################################################################"""
        """ ##################################  Logging Parameters  #####################################"""
        """##############################################################################################"""

        experimentManager.logManager.write("GDM Parameters LOG")
        experimentManager.logManager.write("Number of Epochs: " + str(epochs))
        experimentManager.logManager.write("Number of Contexts: " + str(num_context))
        experimentManager.logManager.write(
            "Activation Thresholds: [" + str(a_threshold[0]) + ", " + str(a_threshold[1]) + "]")
        experimentManager.logManager.write(
            "Habituation Thresholds: [" + str(h_thresholds[0]) + ", " + str(h_thresholds[1]) + "]")
        experimentManager.logManager.write(
            "Episodic lr: [" + str(e_learning_rates[0]) + ", " + str(e_learning_rates[1]) + "]")
        experimentManager.logManager.write(
            "Semantic lr: [" + str(s_learning_rates[0]) + ", " + str(s_learning_rates[1]) + "]")
        experimentManager.logManager.write("Batch Size: " + str(batch_size))
        experimentManager.logManager.write("GDM Parameters LOG")


        """##############################################################################################"""
        """ ############################   Running Training of Memories  ################################"""
        """##############################################################################################"""

        for s in range(0, ds_vectors.shape[0], batch_size):
            print("Training Episodic Regular")
            g_episodic.train_egwr(ds_vectors[s:s + batch_size],
                                  ds_labels[:, s:s + batch_size],
                                  epochs, a_threshold[0], beta, e_learning_rates,
                                  context, hab_threshold=h_thresholds[0], regulated=0)
            e_weights, e_labels = g_episodic.test(ds_vectors[s:s + batch_size], ds_labels[:, s:s + batch_size],
                                                             test_accuracy=True, ret_vecs=True)
            print("Training Semantic Regular")
            g_semantic.train_egwr(e_weights, e_labels,
                                  epochs, a_threshold[1], beta, s_learning_rates,
                                  context=False, hab_threshold=h_thresholds[1], regulated=1)

            """##############################################################################################"""
            """ ############################   Running Pseudo-Replay  #######################################"""
            """##############################################################################################"""
            if train_replay and n_episodes > 0:
                # Replay pseudo-samples
                for r in range(0, replay_weights.shape[0]):
                    print("Training Episodic Replay")
                    """ ############################   Episodic Pseudo-Replay  #######################################"""
                    g_episodic.train_egwr(replay_weights[r], replay_labels[r, :],
                                          epochs//3, a_threshold[0], beta,
                                          e_learning_rates, context=False, hab_threshold=h_thresholds[0], regulated=0)
                    print("Training Semantic Replay")
                    """ ############################   Semantic Pseudo-Replay  #######################################"""
                    g_semantic.train_egwr(replay_weights[r], replay_labels[r],
                                          epochs//3, a_threshold[1], beta,
                                          s_learning_rates, context=False, hab_threshold=h_thresholds[1], regulated=1)

            """##############################################################################################"""
            """ ############################   Generating Pseudo-Samples  ###################################"""
            """##############################################################################################"""
            if train_replay:
                replay_weights, replay_labels = replay_samples(g_episodic, replay_size)
            n_episodes += 1

        """##############################################################################################"""
        """ ############################   Running Imagination  #########################################"""
        """##############################################################################################"""
        if train_imagine:
            # Current Episodic Memory BMUs
            """ ########################  Existing Episodic Memory BMUs  ################################"""
            replay_weights, replay_labels = replay_samples(g_episodic, replay_size)

            """ ############################   Imagining  ###############################################"""

            e_weights, e_labels = g_episodic.test(ds_vectors, ds_labels, ret_vecs=True)
            imagine_sample, imagine_label = CAAE_model.imagine(e_weights, e_labels, order_label_dictionary, 1,
                                                               dataset=args.dataset)

            # Current and Imagined Samples for training.
            imagine_sample = np.vstack([imagine_sample, np.array(
                                                        [replay_weights[r][0] for r in range(0, replay_weights.shape[0])])])
            imagine_label = np.hstack([imagine_label, np.array(
                [replay_labels[r][0][0] for r in range(0, replay_labels.shape[0])]).reshape(1, replay_labels.shape[0])])
            imagine_label = np.vstack([imagine_label, imagine_label])

            # Batch-wise learning/replay for Imagination
            print("Training Episodic Imagine")
            """ ############################   Episodic with Imagination  ################################"""
            g_episodic.train_egwr(imagine_sample, imagine_label,
                                  epochs//2, a_threshold[0], beta,
                                  e_learning_rates, context=False, hab_threshold=h_thresholds[0], regulated=0)

            print("Training Semantic Imagine")
            """ ############################   Semantic with Imagination  ################################"""
            g_semantic.train_egwr(imagine_sample, imagine_label,
                                  epochs//2, a_threshold[1], beta,
                                  s_learning_rates, context=False, hab_threshold=h_thresholds[1], regulated=1)


        """ ############################   Evaluation with only Current Data ##############################"""
        e_weights, e_labels = g_episodic.test(ds_vectors, ds_labels, ret_vecs=True, test_accuracy=True)
        g_semantic.test(e_weights, e_labels, ret_vecs=True, test_accuracy=True)

        # Logging Accuracy on Current/Novel Data
        experimentManager.logManager.write("Accuracy : Episodic: %s" % (g_episodic.test_accuracy[0]))
        experimentManager.logManager.write("Accuracy : Semantic: %s" % (g_semantic.test_accuracy[0]))

        g_episodic_f1_score = f1_score(y_true=ds_labels[0], y_pred=g_episodic.bmus_label[0], average='micro')
        experimentManager.logManager.write("F1 Score: Episodic: %s" % (g_episodic_f1_score))

        g_semantic_f1_score = f1_score(y_true=ds_labels[0], y_pred=g_semantic.bmus_label[0], average='micro')
        experimentManager.logManager.write("F1 Score: Semantic: %s" % (g_semantic_f1_score))

        return (g_episodic, g_semantic), (g_episodic.test_accuracy[0], g_semantic.test_accuracy[0]), \
               (g_episodic_f1_score, g_semantic_f1_score)


def test_GDM(trained_GWRs, test_data, test_labels, labelsize, type, args=None, mlp_classifier=False, CAAE_model=None,
             order_label_dictionary=None, first=False, classifier_model=None):
    if (trained_GWRs[0] is not None) and not mlp_classifier:
        g_episodic, g_semantic = trained_GWRs
        labels = [labelsize,labelsize]

        ds_labels_test = np.zeros((len(labels), len(test_labels)))
        ds_labels_test[0] = test_labels
        ds_labels_test[1] = test_labels

        # Evaluating Episodic Memory Accuracy on Test Data
        g_episodic.test(test_data, ds_labels_test, test_accuracy=True)

        # Evaluating Semantic Memory Accuracy on Test Data
        g_semantic.test(test_data, ds_labels_test, test_accuracy=True)

        experimentManager.logManager.write(type + " Testing Accuracy: Episodic: %s" % (g_episodic.test_accuracy[0]))
        experimentManager.logManager.write(type + " Testing Accuracy: Semantic: %s" % (g_semantic.test_accuracy[0]))
        g_episodic_f1_score = f1_score(y_true=test_labels, y_pred=g_episodic.bmus_label[0], average='micro')
        experimentManager.logManager.write(type + " Testing F1 Score: Episodic: %s" % (g_episodic_f1_score))

        g_semantic_f1_score = f1_score(y_true=test_labels, y_pred=g_semantic.bmus_label[0], average='micro')
        experimentManager.logManager.write(type + " Testing F1 Score: Semantic: %s" % (g_semantic_f1_score))

        return (g_episodic.test_accuracy[0], g_semantic.test_accuracy[0]), (g_episodic_f1_score, g_semantic_f1_score)
    else:
        classifier_model, mlp_accs, f1s = test_mlp(args=args, data=test_data, labels=test_labels, CAAE_model=CAAE_model,
                                 order=order_label_dictionary, first=first, classifier_model=classifier_model)
        return mlp_accs, f1s

def compute_mean_si_dev(data, condition, axis, finalAcc=False):

    if finalAcc:
        mean = np.mean(data[:, condition, :, -1], axis=axis)
        std_err = sem(data[:, condition, :, -1], axis=axis)
        std_dev = np.std(data[:, condition, :, -1], axis=axis)
        values = data[:, condition, :, -1]
    else:
        mean = np.mean(data[:, condition, :, :], axis=axis)
        std_err = sem(data[:, condition, :, :], axis=axis)
        std_dev = np.std(data[:, condition, :, :], axis=axis)
        values = data[:, condition, :, :]
    return mean, std_err, std_dev, values

def compute_stats(experimentManager, class_orders, subjects, class_dictionary, all_seen_acc_GDM_E_all_per_order,
                  all_seen_acc_GDM_S_all_per_order, overall_acc_GDM_E_all_per_order, overall_acc_GDM_S_all_per_order,
                  all_seen_f1_GDM_E_all_per_order, all_seen_f1_GDM_S_all_per_order, overall_f1_GDM_E_all_per_order,
                  overall_f1_GDM_S_all_per_order, args):

    def compute_mean_si_dev(data, condition, axis, finalAcc=False):
        """ Shape of data: (len(class_orders),  len(conditions), len(subjects), args.classes) """

        if finalAcc:
            mean = np.mean(data[:, condition, :, -1], axis=axis)
            std_err = sem(data[:, condition, :, -1], axis=axis)
            std_dev = np.std(data[:, condition, :, -1], axis=axis)
            values = data[:, condition, :, -1]
        else:
            mean = np.mean(data[:, condition, :, :], axis=axis)
            std_err = sem(data[:, condition, :, :], axis=axis)
            std_dev = np.std(data[:, condition, :, :], axis=axis)
            values = data[:, condition, :, :]
        return mean, std_err, std_dev, values

    def compute_statistics(data):
        def kruksalcompute(args):
            return kruskal(*args)
        """ Shape of data: 
            Either data-> (len(orders), len(subjects), len(classes)
            or     data-> (len(subjects), len(orders, len(classes) """
        k_stats = []
        a_stats = []

        for c in range(1,data.shape[2]):
            k_stats.append(kruksalcompute(data[:,:,c].tolist()))
            a_stats.append(anderson_ksamp(data[:,:,c].tolist()))
        return k_stats, a_stats

    """#######################################################################################################"""
    """#################################       Computing stats per-order #####################################"""
    """#######################################################################################################"""

    save_per_order = experimentManager.outputsDirectory + "/" + "Per_Order"
    if not os.path.exists(save_per_order):
        os.makedirs(save_per_order)

    """#######################################################################################################"""
    """##############################       Incremental Learning            ##################################"""
    """#######################################################################################################"""
    """############################## Using Accuracy at each class addition ##################################"""
    """#######################################################################################################"""
    """##############################         Episodic Memory               ##################################"""
    """#######################################################################################################"""

    """#######################################################################################################"""
    """###### Computing mean, Std. Dev and Std. Err across subjects for each order for all condition ##########"""
    """#######################################################################################################"""

    GDM_all_seen_acc_GDM_E_all_per_order_mean, \
    GDM_all_seen_acc_GDM_E_all_per_order_std_err, \
    GDM_all_seen_acc_GDM_E_all_per_order_std_dev, \
    GDM_all_seen_acc_GDM_E_all_per_order_values = compute_mean_si_dev(data=all_seen_acc_GDM_E_all_per_order,
                                                                       condition=0, axis=1)

    GDM_replay_all_seen_acc_GDM_E_all_per_order_mean, \
    GDM_replay_all_seen_acc_GDM_E_all_per_order_std_err, \
    GDM_replay_all_seen_acc_GDM_E_all_per_order_std_dev , \
    GDM_replay_all_seen_acc_GDM_E_all_per_order_values  = compute_mean_si_dev(data=all_seen_acc_GDM_E_all_per_order,
                                                                              condition=1, axis=1)

    GDM_imagine_all_seen_acc_GDM_E_all_per_order_mean, \
    GDM_imagine_all_seen_acc_GDM_E_all_per_order_std_err, \
    GDM_imagine_all_seen_acc_GDM_E_all_per_order_std_dev, \
    GDM_imagine_all_seen_acc_GDM_E_all_per_order_values   = compute_mean_si_dev(data=all_seen_acc_GDM_E_all_per_order,
                                                                               condition=1, axis=1)

    # Reshaping to Conditions, Orders, Subjects, Classes
    all_seen_acc_GDM_E_all_subjects_per_condition_per_order = np.array([all_seen_acc_GDM_E_all_per_order[:, 0, :, :],
                                                     all_seen_acc_GDM_E_all_per_order[:, 1, :, :],
                                                     all_seen_acc_GDM_E_all_per_order[:, 2, :, :]]).reshape((
                                                                                                    3,
                                                                                                    len(class_orders),
                                                                                                    len(subjects),
                                                                                                    args.classes))

    all_seen_f1_GDM_E_all_subjects_per_condition_per_order = np.array([all_seen_f1_GDM_E_all_per_order[:, 0, :, :],
                                                                all_seen_f1_GDM_E_all_per_order[:, 1, :, :],
                                                                all_seen_f1_GDM_E_all_per_order[:, 2, :, :]]).reshape((
                                                                                        3, len(class_orders),
                                                                                        len(subjects), args.classes))

    """#######################################################################################################"""
    """# Computing Kruskal-Wallis H-test and k-sample Anderson-Darling test to compare accuracy at each step #"""
    """#######################################################################################################"""


    """######################################### For GDM Condition  ##########################################"""
    kruskal_stats, anderson_stats = compute_statistics(
        data=np.array(GDM_all_seen_acc_GDM_E_all_per_order_values).reshape(
                                                                    (len(class_orders), len(subjects), args.classes)))
    experimentManager.logManager.write("Per_Order_Incremental Learning GDM-E KRUSKAL TEST:")
    experimentManager.logManager.write(kruskal_stats)

    experimentManager.logManager.write("Per_Order_Incremental Learning GDM-E ANDERSON KSAMP TEST:")
    experimentManager.logManager.write(anderson_stats)

    experimentManager.logManager.write("Plotting Violin/Box Plots for Per_Order_Incremental Learning GDM-E:")
    experimentManager.logManager.write(
        GDM_all_seen_acc_GDM_E_all_per_order_values.reshape((len(class_orders), len(subjects), args.classes)))

    plot_violin(
        data=GDM_all_seen_acc_GDM_E_all_per_order_values.reshape((len(class_orders), len(subjects), args.classes)),
        location=save_per_order,
        incremental=True,
        episodic=True,
        type="Per_Order",
        individual = True,
        condition="GDM-E",
        kruskal_stats=kruskal_stats,
        anderson_stats=anderson_stats)

    """#################################### For GDM + Replay Condition  ######################################"""
    kruskal_stats, anderson_stats = compute_statistics(
        data=np.array(GDM_replay_all_seen_acc_GDM_E_all_per_order_values).reshape(
                                                                    (len(class_orders), len(subjects), args.classes)))

    experimentManager.logManager.write("Per_Order_Incremental Learning GDM-E + Replay KRUSKAL TEST:")
    experimentManager.logManager.write(kruskal_stats)

    experimentManager.logManager.write("Per_Order_Incremental Learning GDM-E + Replay ANDERSON KSAMP TEST:")
    experimentManager.logManager.write(anderson_stats)

    experimentManager.logManager.write("Plotting Violin/Box Plot for Per_Order_Incremental Learning GDM-E + Replay:")
    experimentManager.logManager.write(
        GDM_replay_all_seen_acc_GDM_E_all_per_order_values.reshape((len(class_orders), len(subjects), args.classes)))

    plot_violin(data=GDM_replay_all_seen_acc_GDM_E_all_per_order_values.reshape(
                                                                    (len(class_orders), len(subjects), args.classes)),
                location=save_per_order,
                incremental=True,
                episodic=True,
                type="Per_Order",
                individual=True,
                condition="GDM-E+Replay",
                kruskal_stats=kruskal_stats,
                anderson_stats=anderson_stats)

    """############################### For GDM + Imagination Condition  ######################################"""
    kruskal_stats, anderson_stats = compute_statistics(
        data=np.array(GDM_imagine_all_seen_acc_GDM_E_all_per_order_values).reshape(
                                                                    (len(class_orders), len(subjects), args.classes)))

    experimentManager.logManager.write("Per_Order_Incremental Learning GDM-E + Imagination KRUSKAL TEST:")
    experimentManager.logManager.write(kruskal_stats)

    experimentManager.logManager.write("Per_Order_Incremental Learning GDM-E + Imagination ANDERSON KSAMP TEST:")
    experimentManager.logManager.write(anderson_stats)

    experimentManager.logManager.write("Plotting Violin/Box Plot  Per_Order_Incremental Learning GDM-E + Imagination:")
    experimentManager.logManager.write(
        GDM_imagine_all_seen_acc_GDM_E_all_per_order_values.reshape((len(class_orders), len(subjects), args.classes)))

    plot_violin(data=GDM_imagine_all_seen_acc_GDM_E_all_per_order_values.reshape(
                                                                    (len(class_orders), len(subjects), args.classes)),
                location=save_per_order,
                incremental=True,
                episodic=True,
                type="Per_Order",
                individual=True,
                condition="GDM-E+Imagine",
                kruskal_stats=kruskal_stats,
                anderson_stats=anderson_stats)


    """############################### Plotting differences between 3 conditions #############################"""
    plot_violin(data=np.array([GDM_all_seen_acc_GDM_E_all_per_order_mean,
                               GDM_replay_all_seen_acc_GDM_E_all_per_order_mean,
                               GDM_imagine_all_seen_acc_GDM_E_all_per_order_mean]).reshape(
                                                                                (3,len(class_orders), args.classes)),
                location=save_per_order,
                incremental=True,
                episodic=True,
                type="Per_Order",
                individual=False)

    """#######################################################################################################"""
    """###################################### Semantic Memory ################################################"""
    """#######################################################################################################"""

    """#######################################################################################################"""
    """###### Computing mean, Std. Dev and Std. Err across subjects for each order for all condition ##########"""
    """#######################################################################################################"""

    GDM_all_seen_acc_GDM_S_all_per_order_mean, \
    GDM_all_seen_acc_GDM_S_all_per_order_std_err, \
    GDM_all_seen_acc_GDM_S_all_per_order_std_dev, \
    GDM_all_seen_acc_GDM_S_all_per_order_values = compute_mean_si_dev(data=all_seen_acc_GDM_S_all_per_order,
                                                                       condition=0,
                                                                       axis=1)

    GDM_replay_all_seen_acc_GDM_S_all_per_order_mean, \
    GDM_replay_all_seen_acc_GDM_S_all_per_order_std_err, \
    GDM_replay_all_seen_acc_GDM_S_all_per_order_std_dev , \
    GDM_replay_all_seen_acc_GDM_S_all_per_order_values= compute_mean_si_dev(data=all_seen_acc_GDM_S_all_per_order,
                                                                              condition=1,
                                                                              axis=1)

    GDM_imagine_all_seen_acc_GDM_S_all_per_order_mean, \
    GDM_imagine_all_seen_acc_GDM_S_all_per_order_std_err, \
    GDM_imagine_all_seen_acc_GDM_S_all_per_order_std_dev, \
    GDM_imagine_all_seen_acc_GDM_S_all_per_order_values= compute_mean_si_dev(data=all_seen_acc_GDM_S_all_per_order,
                                                                               condition=1,
                                                                               axis=1)

    # Reshaping to Conditions, Orders, Subjects, Classes
    all_seen_acc_GDM_S_all_subjects_per_condition_per_order = np.array([all_seen_acc_GDM_S_all_per_order[:, 0, :, :],
                                                               all_seen_acc_GDM_S_all_per_order[:, 1, :, :],
                                                               all_seen_acc_GDM_S_all_per_order[:, 2, :, :]]).reshape((
                                                                                                    3,
                                                                                                    len(class_orders),
                                                                                                    len(subjects),
                                                                                                    args.classes))

    all_seen_f1_GDM_S_all_subjects_per_condition_per_order = np.array([all_seen_f1_GDM_S_all_per_order[:, 0, :, :],
                                                                all_seen_f1_GDM_S_all_per_order[:, 1, :, :],
                                                                all_seen_f1_GDM_S_all_per_order[:, 2, :, :]]).reshape((
                                                                                                    3,
                                                                                                    len(class_orders),
                                                                                                    len(subjects),
                                                                                                    args.classes))

    """#######################################################################################################"""
    """# Computing Kruskal-Wallis H-test and k-sample Anderson-Darling test to compare accuracy at each step #"""
    """#######################################################################################################"""

    """######################################### For GDM Condition  ##########################################"""
    kruskal_stats, anderson_stats = compute_statistics(
        data=np.array(GDM_all_seen_acc_GDM_S_all_per_order_values).reshape(
                                                                    (len(class_orders), len(subjects), args.classes)))

    experimentManager.logManager.write("Per_Order_Incremental Learning GDM-S KRUSKAL TEST:")
    experimentManager.logManager.write(kruskal_stats)

    experimentManager.logManager.write("Per_Order_Incremental Learning GDM-S ANDERSON KSAMP TEST:")
    experimentManager.logManager.write(anderson_stats)

    experimentManager.logManager.write("Plotting Violin/Box Plots for Per_Order_Incremental Learning GDM-S:")
    experimentManager.logManager.write(
        GDM_all_seen_acc_GDM_S_all_per_order_values.reshape((len(class_orders), len(subjects), args.classes)))

    plot_violin(data=GDM_all_seen_acc_GDM_S_all_per_order_values.reshape(
                                                                    (len(class_orders), len(subjects), args.classes)),
                location=save_per_order,
                incremental=True,
                episodic=False,
                type="Per_Order",
                individual=True,
                condition="GDM-S",
                kruskal_stats=kruskal_stats,
                anderson_stats=anderson_stats)

    """#################################### For GDM + Replay Condition  #######################################"""
    kruskal_stats, anderson_stats = compute_statistics(
        data=np.array(GDM_replay_all_seen_acc_GDM_S_all_per_order_values).reshape(
                                                                    (len(class_orders), len(subjects), args.classes)))

    experimentManager.logManager.write("Per_Order_Incremental Learning GDM-S + Replay KRUSKAL TEST:")
    experimentManager.logManager.write(kruskal_stats)

    experimentManager.logManager.write("Per_Order_Incremental Learning GDM-S + Replay ANDERSON KSAMP TEST:")
    experimentManager.logManager.write(anderson_stats)

    experimentManager.logManager.write("Plotting Violin/Box Plots for Per_Order_Incremental Learning GDM-S + Replay:")
    experimentManager.logManager.write(
        GDM_replay_all_seen_acc_GDM_S_all_per_order_values.reshape((len(class_orders), len(subjects), args.classes)))

    plot_violin(data=GDM_replay_all_seen_acc_GDM_S_all_per_order_values.reshape(
                                                                    (len(class_orders), len(subjects), args.classes)),
                location=save_per_order,
                incremental=True,
                episodic=False,
                type="Per_Order",
                individual=True,
                condition="GDM-S+Replay",
                kruskal_stats=kruskal_stats,
                anderson_stats=anderson_stats)

    """#################################### For GDM + Imagination Condition  #################################"""

    kruskal_stats, anderson_stats = compute_statistics(
        data=np.array(GDM_imagine_all_seen_acc_GDM_S_all_per_order_values).reshape(
                                                                    (len(class_orders), len(subjects), args.classes)))

    experimentManager.logManager.write("Per_Order_Incremental Learning GDM-S + Imagination KRUSKAL TEST:")
    experimentManager.logManager.write(kruskal_stats)

    experimentManager.logManager.write("Per_Order_Incremental Learning GDM-S + Imagination ANDERSON KSAMP TEST:")
    experimentManager.logManager.write(anderson_stats)

    experimentManager.logManager.write("Plotting Violin/Box Plot Per_Order_Incremental Learning GDM-S + Imagination:")
    experimentManager.logManager.write(
        GDM_imagine_all_seen_acc_GDM_S_all_per_order_values.reshape((len(class_orders), len(subjects), args.classes)))

    plot_violin(data=GDM_imagine_all_seen_acc_GDM_S_all_per_order_values.reshape(
                                                                    (len(class_orders), len(subjects), args.classes)),
                location=save_per_order,
                incremental=True,
                episodic=False,
                type="Per_Order",
                individual=True,
                condition="GDM-S+Imagine",
                kruskal_stats=kruskal_stats,
                anderson_stats=anderson_stats)

    """############################### Plotting differences between 3 conditions #############################"""
    plot_violin(data=np.array([GDM_all_seen_acc_GDM_S_all_per_order_mean,
                               GDM_replay_all_seen_acc_GDM_S_all_per_order_mean,
                               GDM_imagine_all_seen_acc_GDM_S_all_per_order_mean]).reshape(
                                                                                (3,len(class_orders), args.classes)),
                location=save_per_order,
                incremental=True,
                episodic=False,
                type="Per_Order",
                individual=False)

    """#######################################################################################################"""
    """###############################       Overall Accuracy       #########################################"""
    """#######################################################################################################"""
    """############################## Using Accuracy at each class addition ##################################"""
    """#######################################################################################################"""
    """###################################### Episodic Memory ################################################"""
    """#######################################################################################################"""

    """#######################################################################################################"""
    """###### Computing mean, Std. Dev and Std. Err across subjects for each order for all condition ##########"""
    """#######################################################################################################"""
    GDM_overall_acc_GDM_E_all_per_order_mean, \
    GDM_overall_acc_GDM_E_all_per_order_std_err, \
    GDM_overall_acc_GDM_E_all_per_order_std_dev, \
    GDM_overall_acc_GDM_E_all_per_order_values= compute_mean_si_dev(data=overall_acc_GDM_E_all_per_order,
                                                                       condition=0,
                                                                       axis=1)

    GDM_replay_overall_acc_GDM_E_all_per_order_mean, \
    GDM_replay_overall_acc_GDM_E_all_per_order_std_err, \
    GDM_replay_overall_acc_GDM_E_all_per_order_std_dev, \
    GDM_replay_overall_acc_GDM_E_all_per_order_values= compute_mean_si_dev(data=overall_acc_GDM_E_all_per_order,
                                                                             condition=1,
                                                                             axis=1)

    GDM_imagine_overall_acc_GDM_E_all_per_order_mean, \
    GDM_imagine_overall_acc_GDM_E_all_per_order_std_err, \
    GDM_imagine_overall_acc_GDM_E_all_per_order_std_dev, \
    GDM_imagine_overall_acc_GDM_E_all_per_order_values= compute_mean_si_dev(data=overall_acc_GDM_E_all_per_order,
                                                                              condition=2,
                                                                              axis=1)


    # Reshaping to Conditions, Orders, Subjects, Classes
    overall_acc_GDM_E_all_subjects_per_condition_per_order = np.array([overall_acc_GDM_E_all_per_order[:, 0, :, :],
                                                              overall_acc_GDM_E_all_per_order[:, 1, :, :],
                                                              overall_acc_GDM_E_all_per_order[:, 2, :, :]]).reshape((
                                                                                                    3,
                                                                                                    len(class_orders),
                                                                                                    len(subjects),
                                                                                                    args.classes))

    overall_f1_GDM_E_all_subjects_per_condition_per_order = np.array([overall_f1_GDM_E_all_per_order[:, 0, :, :],
                                                               overall_f1_GDM_E_all_per_order[:, 1, :, :],
                                                               overall_f1_GDM_E_all_per_order[:, 2, :,:]]).reshape((
                                                                                                3,
                                                                                                len(class_orders),
                                                                                                len(subjects),
                                                                                                args.classes))

    """#######################################################################################################"""
    """# Computing Kruskal-Wallis H-test and k-sample Anderson-Darling test to compare accuracy at each step #"""
    """#######################################################################################################"""

    """######################################### For GDM Condition  ##########################################"""
    kruskal_stats, anderson_stats = compute_statistics(
        data=np.array(GDM_overall_acc_GDM_E_all_per_order_values).reshape(
                                                                    (len(class_orders), len(subjects), args.classes)))

    experimentManager.logManager.write("Per_Order  Overall Accuracy GDM-E KRUSKAL TEST:")
    experimentManager.logManager.write(kruskal_stats)

    experimentManager.logManager.write("Per_Order Overall Accuracy GDM-E ANDERSON KSAMP TEST:")
    experimentManager.logManager.write(anderson_stats)

    experimentManager.logManager.write("Plotting Violin/Box Plots for Per_Order_Overall Accuracy GDM-E:")
    experimentManager.logManager.write(
        GDM_overall_acc_GDM_E_all_per_order_values.reshape((len(class_orders), len(subjects), args.classes)))

    plot_violin(data=GDM_overall_acc_GDM_E_all_per_order_values.reshape(
                                                                    (len(class_orders), len(subjects), args.classes)),
                location=save_per_order,
                incremental=False,
                episodic=True,
                type="Per_Order",
                individual=True,
                condition="GDM-E",
                kruskal_stats=kruskal_stats,
                anderson_stats=anderson_stats)

    """###################################### For GDM + Replay Condition  #####################################"""

    kruskal_stats, anderson_stats = compute_statistics(
        data=np.array(GDM_replay_overall_acc_GDM_E_all_per_order_values).reshape(
                                                                    (len(class_orders), len(subjects), args.classes)))

    experimentManager.logManager.write("Per_Order Overall Accuracy GDM-E + Replay KRUSKAL TEST:")
    experimentManager.logManager.write(kruskal_stats)

    experimentManager.logManager.write("Per_Order Overall Accuracy GDM-E + Replay ANDERSON KSAMP TEST:")
    experimentManager.logManager.write(anderson_stats)

    experimentManager.logManager.write("Plotting Violin/Box Plots for Per_Order_Overall Accuracy GDM-E + Replay:")
    experimentManager.logManager.write(
        GDM_replay_overall_acc_GDM_E_all_per_order_values.reshape((len(class_orders), len(subjects), args.classes)))

    plot_violin(data=GDM_replay_overall_acc_GDM_E_all_per_order_values.reshape(
                                                                    (len(class_orders), len(subjects), args.classes)),
                location=save_per_order,
                incremental=False,
                episodic=True,
                type="Per_Order",
                individual=True,
                condition="GDM-E+Replay",
                kruskal_stats=kruskal_stats,
                anderson_stats=anderson_stats)

    """################################### For GDM + Imagination Condition  ##################################"""

    kruskal_stats, anderson_stats = compute_statistics(
        data=np.array(GDM_imagine_overall_acc_GDM_E_all_per_order_values).reshape(
                                                                    (len(class_orders), len(subjects), args.classes)))

    experimentManager.logManager.write("Per_Order_Overall Accuracy GDM-E + Imagination KRUSKAL TEST:")
    experimentManager.logManager.write(kruskal_stats)

    experimentManager.logManager.write("Per_Order_Overall Accuracy GDM-E + Imagination ANDERSON KSAMP TEST:")
    experimentManager.logManager.write(anderson_stats)

    experimentManager.logManager.write("Plotting Violin/Box Plots for Per_Order_Overall Accuracy GDM-E + Imagination:")
    experimentManager.logManager.write(
        GDM_imagine_overall_acc_GDM_E_all_per_order_values.reshape((len(class_orders), len(subjects), args.classes)))

    plot_violin(data=GDM_imagine_overall_acc_GDM_E_all_per_order_values.reshape(
                                                                    (len(class_orders), len(subjects), args.classes)),
                location=save_per_order,
                incremental=False,
                episodic=True,
                type="Per_Order",
                individual=True,
                condition="GDM-E+Imagine",
                kruskal_stats=kruskal_stats,
                anderson_stats=anderson_stats)

    """############################### Plotting differences between 3 conditions #############################"""
    plot_violin(data=np.array([GDM_overall_acc_GDM_E_all_per_order_mean,
                               GDM_replay_overall_acc_GDM_E_all_per_order_mean,
                               GDM_imagine_overall_acc_GDM_E_all_per_order_mean]).reshape(
                                                                                (3,len(class_orders), args.classes)),
               location=save_per_order,
               incremental=False,
               episodic=True,
               type="Per_Order", individual=False)

    """#######################################################################################################"""
    """###################################### Semantic Memory ################################################"""
    """#######################################################################################################"""

    """#######################################################################################################"""
    """###### Computing mean, Std. Dev and Std. Err across subjects for each order for all condition ##########"""
    """#######################################################################################################"""
    GDM_overall_acc_GDM_S_all_per_order_mean, \
    GDM_overall_acc_GDM_S_all_per_order_std_err, \
    GDM_overall_acc_GDM_S_all_per_order_std_dev, \
    GDM_overall_acc_GDM_S_all_per_order_values  = compute_mean_si_dev(data=overall_acc_GDM_S_all_per_order,
                                                                      condition=0,
                                                                      axis=1)

    GDM_replay_overall_acc_GDM_S_all_per_order_mean, \
    GDM_replay_overall_acc_GDM_S_all_per_order_std_err, \
    GDM_replay_overall_acc_GDM_S_all_per_order_std_dev, \
    GDM_replay_overall_acc_GDM_S_all_per_order_values= compute_mean_si_dev(data=overall_acc_GDM_S_all_per_order,
                                                                             condition=1,
                                                                             axis=1)

    GDM_imagine_overall_acc_GDM_S_all_per_order_mean, \
    GDM_imagine_overall_acc_GDM_S_all_per_order_std_err, \
    GDM_imagine_overall_acc_GDM_S_all_per_order_std_dev, \
    GDM_imagine_overall_acc_GDM_S_all_per_order_values= compute_mean_si_dev(data=overall_acc_GDM_S_all_per_order,
                                                                              condition=2,
                                                                              axis=1)
    # Reshaping to Conditions, Orders, Subjects, Classes
    overall_acc_GDM_S_all_subjects_per_condition_per_order = np.array([overall_acc_GDM_S_all_per_order[:, 0, :, :],
                                                              overall_acc_GDM_S_all_per_order[:, 1, :, :],
                                                              overall_acc_GDM_S_all_per_order[:, 2, :, :]]).reshape((
                                                                                                    3,
                                                                                                    len(class_orders),
                                                                                                    len(subjects),
                                                                                                    args.classes))

    overall_f1_GDM_S_all_subjects_per_condition_per_order = np.array([overall_f1_GDM_S_all_per_order[:, 0, :, :],
                                                               overall_f1_GDM_S_all_per_order[:, 1, :, :],
                                                               overall_f1_GDM_S_all_per_order[:, 2, :,:]]).reshape((
                                                                                                    3,
                                                                                                    len(class_orders),
                                                                                                    len(subjects),
                                                                                                    args.classes))

    """#######################################################################################################"""
    """# Computing Kruskal-Wallis H-test and k-sample Anderson-Darling test to compare accuracy at each step #"""
    """#######################################################################################################"""

    """######################################### For GDM Condition  ##########################################"""
    kruskal_stats, anderson_stats = compute_statistics(
        data=np.array(GDM_overall_acc_GDM_S_all_per_order_values).reshape(
                                                                    (len(class_orders), len(subjects), args.classes)))

    experimentManager.logManager.write("Per_Order  Overall Accuracy GDM-S KRUSKAL TEST:")
    experimentManager.logManager.write(kruskal_stats)

    experimentManager.logManager.write("Per_Order Overall Accuracy GDM-S ANDERSON KSAMP TEST:")
    experimentManager.logManager.write(anderson_stats)

    experimentManager.logManager.write("Plotting Violin/Box Plots for Per_Order_Overall Accuracy GDM-S:")
    experimentManager.logManager.write(
        GDM_overall_acc_GDM_S_all_per_order_values.reshape((len(class_orders), len(subjects), args.classes)))

    plot_violin(data=GDM_overall_acc_GDM_S_all_per_order_values.reshape(
                                                                    (len(class_orders), len(subjects), args.classes)),
                location=save_per_order,
                incremental=False,
                episodic=False,
                type="Per_Order",
                individual=True,
                condition="GDM-S",
                kruskal_stats=kruskal_stats,
                anderson_stats=anderson_stats)

    """###################################### For GDM + Replay Condition  #####################################"""
    kruskal_stats, anderson_stats = compute_statistics(
        data=np.array(GDM_replay_overall_acc_GDM_S_all_per_order_values).reshape(
                                                                    (len(class_orders), len(subjects), args.classes)))

    experimentManager.logManager.write("Per_Order Overall Accuracy GDM-S + Replay KRUSKAL TEST:")
    experimentManager.logManager.write(kruskal_stats)

    experimentManager.logManager.write("Per_Order Overall Accuracy GDM-S + Replay ANDERSON KSAMP TEST:")
    experimentManager.logManager.write(anderson_stats)

    experimentManager.logManager.write("Plotting Violin/Box Plots for Per_Order_Overall Accuracy GDM-S + Replay:")
    experimentManager.logManager.write(
        GDM_replay_overall_acc_GDM_S_all_per_order_values.reshape((len(class_orders), len(subjects), args.classes)))

    plot_violin(data=GDM_replay_overall_acc_GDM_S_all_per_order_values.reshape(
                                                                    (len(class_orders), len(subjects), args.classes)),
                location=save_per_order,
                incremental=False,
                episodic=False,
                type="Per_Order",
                individual=True,
                condition="GDM-S+Replay",
                kruskal_stats=kruskal_stats,
                anderson_stats=anderson_stats)


    """################################### For GDM + Imagination Condition  ##################################"""
    kruskal_stats, anderson_stats = compute_statistics(
        data=np.array(GDM_imagine_overall_acc_GDM_S_all_per_order_values).reshape(
                                                                    (len(class_orders), len(subjects), args.classes)))

    experimentManager.logManager.write("Per_Order_Overall Accuracy GDM-S + Imagination KRUSKAL TEST:")
    experimentManager.logManager.write(kruskal_stats)

    experimentManager.logManager.write("Per_Order_Overall Accuracy GDM-S + Imagination ANDERSON KSAMP TEST:")
    experimentManager.logManager.write(anderson_stats)

    experimentManager.logManager.write("Plotting Violin/Box Plots for Per_Order_Overall Accuracy GDM-S + Imagination:")
    experimentManager.logManager.write(
        GDM_imagine_overall_acc_GDM_S_all_per_order_values.reshape((len(class_orders), len(subjects), args.classes)))

    plot_violin(data=GDM_imagine_overall_acc_GDM_S_all_per_order_values.reshape(
                                                                    (len(class_orders), len(subjects), args.classes)),
                location=save_per_order,
                incremental=False,
                episodic=False,
                type="Per_Order",
                individual=True,
                condition="GDM-S+Imagine",
                kruskal_stats=kruskal_stats,
                anderson_stats=anderson_stats)

    """############################### Plotting differences between 3 conditions #############################"""
    plot_violin(data=np.array([GDM_overall_acc_GDM_S_all_per_order_mean,
                               GDM_replay_overall_acc_GDM_S_all_per_order_mean,
                               GDM_imagine_overall_acc_GDM_S_all_per_order_mean]).reshape(
                                                                                (3,len(class_orders), args.classes)),
                location=save_per_order,
                incremental=False,
                episodic=False,
                type="Per_Order",
                individual=False)

    """#######################################################################################################"""
    """###################################### Plot Individual Order plots ####################################"""
    """#######################################################################################################"""

    save_per_order_individual = experimentManager.outputsDirectory + "/Per_Order/IndividualOrders/"
    if not os.path.exists(save_per_order_individual):
        os.makedirs(save_per_order_individual)

    for o in range(len(class_orders)):
        effective_loaded_labels = [class_dictionary.get(i, i) for i in class_orders[o]]
        order_label_dictionary = dict(list(enumerate(effective_loaded_labels)))
        plotting_scores(experimentManager,all_seen_acc_GDM_E_all_subjects_per_condition_per_order[:,o],
                    all_seen_acc_GDM_S_all_subjects_per_condition_per_order[:,o],
                    overall_acc_GDM_E_all_subjects_per_condition_per_order[:,o],
                    overall_acc_GDM_S_all_subjects_per_condition_per_order[:,o],
                        order=order_label_dictionary,location=save_per_order_individual)

        plotting_scores(experimentManager, all_seen_f1_GDM_E_all_subjects_per_condition_per_order[:, o],
                        all_seen_f1_GDM_S_all_subjects_per_condition_per_order[:, o],
                        overall_f1_GDM_E_all_subjects_per_condition_per_order[:, o],
                        overall_f1_GDM_S_all_subjects_per_condition_per_order[:, o],
                        order=order_label_dictionary, location=save_per_order_individual, f1=True)

    """#######################################################################################################"""
    """#################################       Computing stats per-subject ###################################"""
    """#######################################################################################################"""

    save_per_subject = experimentManager.outputsDirectory + "/" + "Per_Subject/"
    if not os.path.exists(save_per_subject):
        os.makedirs(save_per_subject)


    """#######################################################################################################"""
    """##############################       Incremental Learning            ##################################"""
    """#######################################################################################################"""
    """############################## Using Accuracy at each class addition ##################################"""
    """#######################################################################################################"""
    """##############################         Episodic Memory               ##################################"""
    """#######################################################################################################"""

    """#######################################################################################################"""
    """###### Computing mean, Std. Dev and Std. Err across orders for each subject for all conditions ########"""
    """#######################################################################################################"""

    GDM_all_seen_acc_GDM_E_all_per_subject_mean, \
    GDM_all_seen_acc_GDM_E_all_per_subject_std_err, \
    GDM_all_seen_acc_GDM_E_all_per_subject_std_dev, \
    GDM_all_seen_acc_GDM_E_all_per_subject_values  = compute_mean_si_dev(data=all_seen_acc_GDM_E_all_per_order,
                                                                              condition=0,
                                                                              axis=0)

    GDM_replay_all_seen_acc_GDM_E_all_per_subject_mean, \
    GDM_replay_all_seen_acc_GDM_E_all_per_subject_std_err, \
    GDM_replay_all_seen_acc_GDM_E_all_per_subject_std_dev, \
    GDM_replay_all_seen_acc_GDM_E_all_per_subject_values= compute_mean_si_dev(data=all_seen_acc_GDM_E_all_per_order,
                                                                        condition=1,
                                                                        axis=0)

    GDM_imagine_all_seen_acc_GDM_E_all_per_subject_mean, \
    GDM_imagine_all_seen_acc_GDM_E_all_per_subject_std_err, \
    GDM_imagine_all_seen_acc_GDM_E_all_per_subject_std_dev, \
    GDM_imagine_all_seen_acc_GDM_E_all_per_subject_values= compute_mean_si_dev(data=all_seen_acc_GDM_E_all_per_order,
                                                                        condition=2,
                                                                        axis=0)
    # Reshaping to Conditions, Orders, Subjects, Classes
    all_seen_acc_GDM_E_all_subjects_per_condition_per_subject = np.array([all_seen_acc_GDM_E_all_per_order[:, 0, :, :],
                                                                all_seen_acc_GDM_E_all_per_order[:, 1, :, :],
                                                                all_seen_acc_GDM_E_all_per_order[:, 2, :, :]]).reshape((
                                                                                                    3,
                                                                                                    len(class_orders),
                                                                                                    len(subjects),
                                                                                                    args.classes))

    all_seen_f1_GDM_E_all_subjects_per_condition_per_subject = np.array([all_seen_f1_GDM_E_all_per_order[:, 0, :, :],
                                                                          all_seen_f1_GDM_E_all_per_order[:, 1, :, :],
                                                                          all_seen_f1_GDM_E_all_per_order[:, 2, :,
                                                                          :]]).reshape((
        3,
        len(class_orders),
        len(subjects),
        args.classes))

    """#######################################################################################################"""
    """# Computing Kruskal-Wallis H-test and k-sample Anderson-Darling test to compare accuracy at each step #"""
    """#######################################################################################################"""

    """######################################### For GDM Condition  ##########################################"""
    kruskal_stats, anderson_stats = compute_statistics(
        data=np.array(GDM_all_seen_acc_GDM_E_all_per_subject_values).reshape(
                                                                    (len(class_orders), len(subjects), args.classes)))

    experimentManager.logManager.write("Per_subject Incremental Learning GDM-E KRUSKAL TEST:")
    experimentManager.logManager.write(kruskal_stats)

    experimentManager.logManager.write("Per_subject Incremental Learning GDM-E ANDERSON KSAMP TEST:")
    experimentManager.logManager.write(anderson_stats)

    experimentManager.logManager.write("Plotting Violin/Box Plots for Per_subject Incremental Learning GDM-E")
    experimentManager.logManager.write(
        GDM_all_seen_acc_GDM_E_all_per_subject_values.reshape((len(class_orders), len(subjects), args.classes)))

    plot_violin(data=np.transpose(a=GDM_all_seen_acc_GDM_E_all_per_subject_values.reshape(
                                                    (len(class_orders), len(subjects), args.classes)),
                                  axes=(1, 0, 2)),
                location=save_per_order,
                incremental=True,
                episodic=True,
                type="Per_subject",
                individual=True,
                condition="GDM-E",
                kruskal_stats=kruskal_stats,
                anderson_stats=anderson_stats)

    """#################################### For GDM + Replay Condition  ######################################"""
    kruskal_stats, anderson_stats = compute_statistics(
        data=np.array(GDM_replay_all_seen_acc_GDM_E_all_per_subject_values).reshape(
                                                                    (len(class_orders), len(subjects), args.classes)))

    experimentManager.logManager.write("Per_subject Incremental Learning GDM-E + Replay KRUSKAL TEST:")
    experimentManager.logManager.write(kruskal_stats)

    experimentManager.logManager.write("Per_subject Incremental Learning GDM-E + Replay ANDERSON KSAMP TEST:")
    experimentManager.logManager.write(anderson_stats)

    experimentManager.logManager.write("Plotting Violin/Box Plots for Per_subject Incremental Learning GDM-E + Replay")
    experimentManager.logManager.write(
        GDM_replay_all_seen_acc_GDM_E_all_per_subject_values.reshape((len(class_orders), len(subjects), args.classes)))

    plot_violin(data=np.transpose(a=GDM_replay_all_seen_acc_GDM_E_all_per_subject_values.reshape(
                                                    (len(class_orders), len(subjects), args.classes)),
                                  axes=(1, 0, 2)),
                location=save_per_order,
                incremental=True,
                episodic=True,
                type="Per_subject",
                individual=True,
                condition="GDM-E+Replay",
                kruskal_stats=kruskal_stats,
                anderson_stats=anderson_stats)

    """############################### For GDM + Imagination Condition  ######################################"""
    kruskal_stats, anderson_stats = compute_statistics(
        data=np.array(GDM_imagine_all_seen_acc_GDM_E_all_per_subject_values).reshape(
                                                                    (len(class_orders), len(subjects), args.classes)))

    experimentManager.logManager.write("Per_subject Incremental Learning GDM-E + Imagination KRUSKAL TEST:")
    experimentManager.logManager.write(kruskal_stats)

    experimentManager.logManager.write("Per_subject Incremental Learning GDM-E + Imagination ANDERSON KSAMP TEST:")
    experimentManager.logManager.write(anderson_stats)

    experimentManager.logManager.write("Plotting Violin/Box Plot Per_subject Incremental Learning GDM-E + Imagination")
    experimentManager.logManager.write(
        GDM_imagine_all_seen_acc_GDM_E_all_per_subject_values.reshape((len(class_orders), len(subjects), args.classes)))

    plot_violin(data=np.transpose(a=GDM_imagine_all_seen_acc_GDM_E_all_per_subject_values.reshape(
                                                    (len(class_orders), len(subjects), args.classes)),
                                  axes=(1, 0, 2)),
                location=save_per_order,
                incremental=True,
                episodic=True,
                type="Per_subject",
                individual=True,
                condition="GDM-E+Imagine",
                kruskal_stats=kruskal_stats,
                anderson_stats=anderson_stats)

    """############################### Plotting differences between 3 conditions #############################"""
    plot_violin(data=np.array([GDM_all_seen_acc_GDM_E_all_per_subject_mean,
                               GDM_replay_all_seen_acc_GDM_E_all_per_subject_mean,
                               GDM_imagine_all_seen_acc_GDM_E_all_per_subject_mean]).reshape((3, len(subjects),
                                                                                            args.classes)),
                location=save_per_subject,
                incremental=True,
                episodic=True,
                type="Per_subject",
                individual=False)

    """#######################################################################################################"""
    """###################################### Semantic Memory ################################################"""
    """#######################################################################################################"""

    """#######################################################################################################"""
    """###### Computing mean, Std. Dev and Std. Err across orders for each subject for all conditions ########"""
    """#######################################################################################################"""

    GDM_all_seen_acc_GDM_S_all_per_subject_mean, \
    GDM_all_seen_acc_GDM_S_all_per_subject_std_err, \
    GDM_all_seen_acc_GDM_S_all_per_subject_std_dev, \
    GDM_all_seen_acc_GDM_S_all_per_subject_values= compute_mean_si_dev(data=all_seen_acc_GDM_S_all_per_order,
                                                                        condition=0,
                                                                        axis=0)

    GDM_replay_all_seen_acc_GDM_S_all_per_subject_mean, \
    GDM_replay_all_seen_acc_GDM_S_all_per_subject_std_err, \
    GDM_replay_all_seen_acc_GDM_S_all_per_subject_std_dev, \
    GDM_replay_all_seen_acc_GDM_S_all_per_subject_values= compute_mean_si_dev(data=all_seen_acc_GDM_S_all_per_order,
                                                                               condition=1,
                                                                               axis=0)

    GDM_imagine_all_seen_acc_GDM_S_all_per_subject_mean, \
    GDM_imagine_all_seen_acc_GDM_S_all_per_subject_std_err, \
    GDM_imagine_all_seen_acc_GDM_S_all_per_subject_std_dev, \
    GDM_imagine_all_seen_acc_GDM_S_all_per_subject_values= compute_mean_si_dev(data=all_seen_acc_GDM_S_all_per_order,
                                                                                condition=2,
                                                                                axis=0)
    # Reshaping to Conditions, Orders, Subjects, Classes
    all_seen_acc_GDM_S_all_subjects_per_condition_per_subject = np.array([all_seen_acc_GDM_S_all_per_order[:, 0, :, :],
                                                                all_seen_acc_GDM_S_all_per_order[:, 1, :, :],
                                                                all_seen_acc_GDM_S_all_per_order[:, 2, :, :]]).reshape((
                                                                                                    3,
                                                                                                    len(class_orders),
                                                                                                    len(subjects),
                                                                                                    args.classes))

    all_seen_f1_GDM_S_all_subjects_per_condition_per_subject = np.array([all_seen_f1_GDM_S_all_per_order[:, 0, :, :],
                                                                          all_seen_f1_GDM_S_all_per_order[:, 1, :, :],
                                                                          all_seen_f1_GDM_S_all_per_order[:, 2, :,
                                                                          :]]).reshape((
        3,
        len(class_orders),
        len(subjects),
        args.classes))

    """#######################################################################################################"""
    """# Computing Kruskal-Wallis H-test and k-sample Anderson-Darling test to compare accuracy at each step #"""
    """#######################################################################################################"""

    """######################################### For GDM Condition  ##########################################"""
    kruskal_stats, anderson_stats = compute_statistics(
        data=np.array(GDM_all_seen_acc_GDM_S_all_per_subject_values).reshape(
                                                                    (len(class_orders), len(subjects), args.classes)))

    experimentManager.logManager.write("Per_subject Incremental Learning GDM-S KRUSKAL TEST:")
    experimentManager.logManager.write(kruskal_stats)

    experimentManager.logManager.write("Per_subject Incremental Learning GDM-S ANDERSON KSAMP TEST:")
    experimentManager.logManager.write(anderson_stats)

    experimentManager.logManager.write("Plotting Violin/Box Plots for Per_subject Incremental Learning GDM-S")
    experimentManager.logManager.write(
        GDM_all_seen_acc_GDM_S_all_per_subject_values.reshape((len(class_orders), len(subjects), args.classes)))

    plot_violin(data=np.transpose(a=GDM_all_seen_acc_GDM_S_all_per_subject_values.reshape(
                                                    (len(class_orders), len(subjects), args.classes)),
                                  axes=(1, 0, 2)),
                location=save_per_order,
                incremental=True,
                episodic=False,
                type="Per_subject",
                individual=True,
                condition="GDM-S",
                kruskal_stats=kruskal_stats,
                anderson_stats=anderson_stats)

    """#################################### For GDM + Replay Condition  ######################################"""
    kruskal_stats, anderson_stats = compute_statistics(
        data=np.array(GDM_replay_all_seen_acc_GDM_S_all_per_subject_values).reshape(
                                                                    (len(class_orders), len(subjects), args.classes)))

    experimentManager.logManager.write("Per_subject Incremental Learning GDM-S + Replay KRUSKAL TEST:")
    experimentManager.logManager.write(kruskal_stats)

    experimentManager.logManager.write("Per_subject Incremental Learning GDM-S + Replay ANDERSON KSAMP TEST:")
    experimentManager.logManager.write(anderson_stats)

    experimentManager.logManager.write("Plotting Violin/Box Plots for Per_subject Incremental Learning GDM-S + Replay")
    experimentManager.logManager.write(
        GDM_replay_all_seen_acc_GDM_S_all_per_subject_values.reshape((len(class_orders), len(subjects), args.classes)))

    plot_violin(data=np.transpose(a=GDM_replay_all_seen_acc_GDM_S_all_per_subject_values.reshape(
                                                    (len(class_orders), len(subjects), args.classes)),
                                  axes=(1, 0, 2)),
                location=save_per_order,
                incremental=True,
                episodic=False,
                type="Per_subject",
                individual=True,
                condition="GDM-S+Replay",
                kruskal_stats=kruskal_stats,
                anderson_stats=anderson_stats)

    """############################### For GDM + Imagination Condition  ######################################"""
    kruskal_stats, anderson_stats = compute_statistics(
        data=np.array(GDM_imagine_all_seen_acc_GDM_S_all_per_subject_values).reshape(
                                                                    (len(class_orders), len(subjects), args.classes)))

    experimentManager.logManager.write("Per_subject Incremental Learning GDM-S + Imagination KRUSKAL TEST:")
    experimentManager.logManager.write(kruskal_stats)

    experimentManager.logManager.write("Per_subject Incremental Learning GDM-S + Imagination ANDERSON KSAMP TEST:")
    experimentManager.logManager.write(anderson_stats)

    experimentManager.logManager.write("Plotting Violin/Box Plot Per_subject Incremental Learning GDM-S + Imagination")
    experimentManager.logManager.write(
        GDM_imagine_all_seen_acc_GDM_S_all_per_subject_values.reshape((len(class_orders), len(subjects), args.classes)))

    plot_violin(data=np.transpose(a=GDM_imagine_all_seen_acc_GDM_S_all_per_subject_values.reshape(
                                                    (len(class_orders), len(subjects), args.classes)),
                                  axes=(1, 0, 2)),
                location=save_per_order,
                incremental=True,
                episodic=False,
                type="Per_subject",
                individual=True,
                condition="GDM-S+Imagine",
                kruskal_stats=kruskal_stats,
                anderson_stats=anderson_stats)

    """############################### Plotting differences between 3 conditions #############################"""
    plot_violin(data=np.array([GDM_all_seen_acc_GDM_S_all_per_subject_mean,
                               GDM_replay_all_seen_acc_GDM_S_all_per_subject_mean,
                               GDM_imagine_all_seen_acc_GDM_S_all_per_subject_mean]).reshape((3, len(subjects),
                                                                                             args.classes)),
                location=save_per_subject,
                incremental=True,
                episodic=False,
                type="Per_subject",
                individual=False)

    """#######################################################################################################"""
    """###############################       Overall Accuracy       #########################################"""
    """#######################################################################################################"""
    """############################## Using Accuracy at each class addition ##################################"""
    """#######################################################################################################"""
    """###################################### Episodic Memory ################################################"""
    """#######################################################################################################"""

    """#######################################################################################################"""
    """###### Computing mean, Std. Dev and Std. Err across orders for each subject for all conditions ########"""
    """#######################################################################################################"""
    GDM_overall_acc_GDM_E_all_per_subject_mean, \
    GDM_overall_acc_GDM_E_all_per_subject_std_err, \
    GDM_overall_acc_GDM_E_all_per_subject_std_dev, \
    GDM_overall_acc_GDM_E_all_per_subject_values   = compute_mean_si_dev(data=overall_acc_GDM_E_all_per_order,
                                                                        condition=0,
                                                                        axis=0)
    GDM_replay_overall_acc_GDM_E_all_per_subject_mean, \
    GDM_replay_overall_acc_GDM_E_all_per_subject_std_err, \
    GDM_replay_overall_acc_GDM_E_all_per_subject_std_dev, \
    GDM_replay_overall_acc_GDM_E_all_per_subject_values = compute_mean_si_dev(data=overall_acc_GDM_E_all_per_order,
                                                                        condition=1,
                                                                        axis=0)
    GDM_imagine_overall_acc_GDM_E_all_per_subject_mean, \
    GDM_imagine_overall_acc_GDM_E_all_per_subject_std_err, \
    GDM_imagine_overall_acc_GDM_E_all_per_subject_std_dev, \
    GDM_imagine_overall_acc_GDM_E_all_per_subject_values= compute_mean_si_dev(data=overall_acc_GDM_E_all_per_order,
                                                                        condition=2,
                                                                        axis=0)

    # Reshaping to Conditions, Orders, Subjects, Classes
    overall_acc_GDM_E_all_subjects_per_condition_per_subject = np.array([overall_acc_GDM_E_all_per_order[:, 0, :, :],
                                                               overall_acc_GDM_E_all_per_order[:, 1, :, :],
                                                               overall_acc_GDM_E_all_per_order[:, 2, :, :]]).reshape((
                                                                                                    3,
                                                                                                    len(class_orders),
                                                                                                    len(subjects),
                                                                                                    args.classes))

    overall_f1_GDM_E_all_subjects_per_condition_per_subject = np.array([overall_f1_GDM_E_all_per_order[:, 0, :, :],
                                                                         overall_f1_GDM_E_all_per_order[:, 1, :, :],
                                                                         overall_f1_GDM_E_all_per_order[:, 2, :,
                                                                         :]]).reshape((
        3,
        len(class_orders),
        len(subjects),
        args.classes))

    """#######################################################################################################"""
    """# Computing Kruskal-Wallis H-test and k-sample Anderson-Darling test to compare accuracy at each step #"""
    """#######################################################################################################"""

    """######################################### For GDM Condition  ##########################################"""
    kruskal_stats, anderson_stats = compute_statistics(
        data=np.array(GDM_overall_acc_GDM_E_all_per_subject_values).reshape(
                                                    (len(class_orders), len(subjects), args.classes)))

    experimentManager.logManager.write("Per_subject Overall Accuracy GDM-E KRUSKAL TEST:")
    experimentManager.logManager.write(kruskal_stats)

    experimentManager.logManager.write("Per_subject Overall Accuracy GDM-E ANDERSON KSAMP TEST:")
    experimentManager.logManager.write(anderson_stats)

    experimentManager.logManager.write("Plotting Violin/Box Plots for Per_subject  Overall Accuracy GDM-E")
    experimentManager.logManager.write(
        GDM_overall_acc_GDM_E_all_per_subject_values.reshape((len(class_orders), len(subjects), args.classes)))

    plot_violin(data=np.transpose(a=GDM_overall_acc_GDM_E_all_per_subject_values.reshape(
                                                    (len(class_orders), len(subjects), args.classes)),
                                  axes=(1, 0, 2)),
                location=save_per_order,
                incremental=False,
                episodic=True,
                type="Per_subject",
                individual=True,
                condition="GDM-E",
                kruskal_stats=kruskal_stats,
                anderson_stats=anderson_stats)

    """#################################### For GDM + Replay Condition  ######################################"""
    kruskal_stats, anderson_stats = compute_statistics(
        data=np.array(GDM_replay_overall_acc_GDM_E_all_per_subject_values).reshape(
                                                    (len(class_orders), len(subjects), args.classes)))

    experimentManager.logManager.write("Per_subject Overall Accuracy GDM-E + Replay KRUSKAL TEST:")
    experimentManager.logManager.write(kruskal_stats)

    experimentManager.logManager.write("Per_subject Overall Accuracy GDM-E + Replay ANDERSON KSAMP TEST:")
    experimentManager.logManager.write(anderson_stats)

    experimentManager.logManager.write("Plotting Violin/Box Plots for Per_subject  Overall Accuracy GDM-E + Replay")
    experimentManager.logManager.write(
        GDM_replay_overall_acc_GDM_E_all_per_subject_values.reshape((len(class_orders), len(subjects), args.classes)))

    plot_violin(data=np.transpose(a=GDM_replay_overall_acc_GDM_E_all_per_subject_values.reshape(
                                                    (len(class_orders), len(subjects), args.classes)),
                                  axes=(1, 0, 2)),
                location=save_per_order,
                incremental=False,
                episodic=True,
                type="Per_subject",
                individual=True,
                condition="GDM-E+Replay",
                kruskal_stats=kruskal_stats,
                anderson_stats=anderson_stats)

    """############################### For GDM + Imagination Condition  ######################################"""
    kruskal_stats, anderson_stats = compute_statistics(
        data=np.array(GDM_imagine_overall_acc_GDM_E_all_per_subject_values).reshape(
                                                    (len(class_orders), len(subjects), args.classes)))

    experimentManager.logManager.write("Per_subject Overall Accuracy GDM-E + Imagination KRUSKAL TEST:")
    experimentManager.logManager.write(kruskal_stats)

    experimentManager.logManager.write("Per_subject Overall Accuracy GDM-E + Imagination ANDERSON KSAMP TEST:")
    experimentManager.logManager.write(anderson_stats)

    experimentManager.logManager.write("Plotting Violin/Box Plot for Per_subject  Overall Accuracy GDM-E + Imagination")
    experimentManager.logManager.write(
        GDM_imagine_overall_acc_GDM_E_all_per_subject_values.reshape((len(class_orders), len(subjects), args.classes)))

    plot_violin(data=np.transpose(a=GDM_imagine_overall_acc_GDM_E_all_per_subject_values.reshape(
                                                    (len(class_orders), len(subjects), args.classes)),
                                  axes=(1, 0, 2)),
                location=save_per_order,
                incremental=False,
                episodic=True,
                type="Per_subject",
                individual=True,
                condition="GDM-E+Imagine",
                kruskal_stats=kruskal_stats,
                anderson_stats=anderson_stats)

    """############################### Plotting differences between 3 conditions #############################"""
    plot_violin(data=np.array([GDM_overall_acc_GDM_E_all_per_subject_mean,
                               GDM_replay_overall_acc_GDM_E_all_per_subject_mean,
                               GDM_imagine_overall_acc_GDM_E_all_per_subject_mean]).reshape((3, len(subjects),
                                                                                            args.classes)),
                location=save_per_subject,
                incremental=False,
                episodic=True,
                type="Per_subject",
                individual=False)

    """#######################################################################################################"""
    """###################################### Semantic Memory ################################################"""
    """#######################################################################################################"""

    """#######################################################################################################"""
    """###### Computing mean, Std. Dev and Std. Err across orders for each subject for all conditions ########"""
    """#######################################################################################################"""

    GDM_overall_acc_GDM_S_all_per_subject_mean, \
    GDM_overall_acc_GDM_S_all_per_subject_std_err, \
    GDM_overall_acc_GDM_S_all_per_subject_std_dev, \
    GDM_overall_acc_GDM_S_all_per_subject_values= compute_mean_si_dev(data=overall_acc_GDM_S_all_per_order,
                                                                       condition=0,
                                                                       axis=0)
    GDM_replay_overall_acc_GDM_S_all_per_subject_mean, \
    GDM_replay_overall_acc_GDM_S_all_per_subject_std_err, \
    GDM_replay_overall_acc_GDM_S_all_per_subject_std_dev, \
    GDM_replay_overall_acc_GDM_S_all_per_subject_values= compute_mean_si_dev(data=overall_acc_GDM_S_all_per_order,
                                                                              condition=1,
                                                                              axis=0)
    GDM_imagine_overall_acc_GDM_S_all_per_subject_mean, \
    GDM_imagine_overall_acc_GDM_S_all_per_subject_std_err, \
    GDM_imagine_overall_acc_GDM_S_all_per_subject_std_dev, \
    GDM_imagine_overall_acc_GDM_S_all_per_subject_values= compute_mean_si_dev(data=overall_acc_GDM_S_all_per_order,
                                                                               condition=2,
                                                                               axis=0)
    # Reshaping to Conditions, Orders, Subjects, Classes

    overall_acc_GDM_S_all_subjects_per_condition_per_subject = np.array([overall_acc_GDM_S_all_per_order[:, 0, :, :],
                                                               overall_acc_GDM_S_all_per_order[:, 1, :, :],
                                                               overall_acc_GDM_S_all_per_order[:, 2, :, :]]).reshape((
                                                                                                    3,
                                                                                                    len(class_orders),
                                                                                                    len(subjects),
                                                                                                    args.classes))

    overall_f1_GDM_S_all_subjects_per_condition_per_subject = np.array([overall_f1_GDM_S_all_per_order[:, 0, :, :],
                                                                         overall_f1_GDM_S_all_per_order[:, 1, :, :],
                                                                         overall_f1_GDM_S_all_per_order[:, 2, :,
                                                                         :]]).reshape((
        3,
        len(class_orders),
        len(subjects),
        args.classes))
    """#######################################################################################################"""
    """# Computing Kruskal-Wallis H-test and k-sample Anderson-Darling test to compare accuracy at each step #"""
    """#######################################################################################################"""

    """######################################### For GDM Condition  ##########################################"""
    kruskal_stats, anderson_stats = compute_statistics(
        data=np.array(GDM_overall_acc_GDM_S_all_per_subject_values).reshape(
                                                    (len(class_orders), len(subjects), args.classes)))

    experimentManager.logManager.write("Per_subject Overall Accuracy GDM-S KRUSKAL TEST:")
    experimentManager.logManager.write(kruskal_stats)

    experimentManager.logManager.write("Per_subject Overall Accuracy GDM-S ANDERSON KSAMP TEST:")
    experimentManager.logManager.write(anderson_stats)

    experimentManager.logManager.write("Plotting Violin/Box Plots for Per_subject  Overall Accuracy GDM-S")
    experimentManager.logManager.write(
        GDM_overall_acc_GDM_S_all_per_subject_values.reshape((len(class_orders), len(subjects), args.classes)))

    plot_violin(data=np.transpose(a=GDM_overall_acc_GDM_S_all_per_subject_values.reshape(
                                                    (len(class_orders), len(subjects), args.classes)),
                                  axes=(1, 0, 2)),
                location=save_per_order,
                incremental=False,
                episodic=False,
                type="Per_subject",
                individual=True,
                condition="GDM-S",
                kruskal_stats=kruskal_stats,
                anderson_stats=anderson_stats)

    """#################################### For GDM + Replay Condition  ######################################"""
    kruskal_stats, anderson_stats = compute_statistics(
        data=np.array(GDM_replay_overall_acc_GDM_S_all_per_subject_values).reshape(
                                                    (len(class_orders), len(subjects), args.classes)))

    experimentManager.logManager.write("Per_subject Overall Accuracy GDM-S + Replay KRUSKAL TEST:")
    experimentManager.logManager.write(kruskal_stats)

    experimentManager.logManager.write("Per_subject Overall Accuracy GDM-S + Replay ANDERSON KSAMP TEST:")
    experimentManager.logManager.write(anderson_stats)

    experimentManager.logManager.write("Plotting Violin/Box Plots for Per_subject  Overall Accuracy GDM-S + Replay:")
    experimentManager.logManager.write(
        GDM_replay_overall_acc_GDM_S_all_per_subject_values.reshape((len(class_orders), len(subjects), args.classes)))

    plot_violin(data=np.transpose(a=GDM_replay_overall_acc_GDM_S_all_per_subject_values.reshape(
                                                    (len(class_orders), len(subjects), args.classes)),
                                  axes=(1, 0, 2)),
                location=save_per_order,
                incremental=False,
                episodic=False,
                type="Per_subject",
                individual=True,
                condition="GDM-S+Replay",
                kruskal_stats=kruskal_stats,
                anderson_stats=anderson_stats)

    """############################### For GDM + Imagination Condition  ######################################"""
    kruskal_stats, anderson_stats = compute_statistics(
        data=np.array(GDM_imagine_overall_acc_GDM_S_all_per_subject_values).reshape(
                                                    (len(class_orders), len(subjects), args.classes)))

    experimentManager.logManager.write("Per_subject Overall Accuracy GDM-S + Imagination KRUSKAL TEST:")
    experimentManager.logManager.write(kruskal_stats)

    experimentManager.logManager.write("Per_subject Overall Accuracy GDM-S + Imagination ANDERSON KSAMP TEST:")
    experimentManager.logManager.write(anderson_stats)

    experimentManager.logManager.write("Plotting Violin/Box Plots for Per_subject  Overall Accuracy GDM-S + Imagination:")
    experimentManager.logManager.write(
        GDM_imagine_overall_acc_GDM_S_all_per_subject_values.reshape((len(class_orders), len(subjects), args.classes)))

    plot_violin(data=np.transpose(a=GDM_imagine_overall_acc_GDM_S_all_per_subject_values.reshape(
                                                    (len(class_orders), len(subjects), args.classes)),
                                  axes=(1, 0, 2)),
                location=save_per_order,
                incremental=False,
                episodic=False,
                type="Per_subject",
                individual=True,
                condition="GDM-S+Imagine",
                kruskal_stats=kruskal_stats,
                anderson_stats=anderson_stats)

    """############################### Plotting differences between 3 conditions #############################"""
    plot_violin(data=np.array([GDM_overall_acc_GDM_S_all_per_subject_mean,
                               GDM_replay_overall_acc_GDM_S_all_per_subject_mean,
                               GDM_imagine_overall_acc_GDM_S_all_per_subject_mean]).reshape((3, len(subjects),
                                                                                            args.classes)),
                location=save_per_subject,
                incremental=False,
                episodic=False,
                type="Per_subject",
                individual=False)

    """#######################################################################################################"""
    """###################################### Plot Individual subject plots ###################################"""
    """#######################################################################################################"""
    save_per_subject_individual = experimentManager.outputsDirectory + "/Per_Subject/IndividualSubjects/"
    if not os.path.exists(save_per_subject_individual):
        os.makedirs(save_per_subject_individual)
    for p in range(len(subjects)):

        plotting_scores(experimentManager,all_seen_acc_GDM_E_all_subjects_per_condition_per_subject[:,:,p],
                        all_seen_acc_GDM_S_all_subjects_per_condition_per_subject[:,:,p],
                        overall_acc_GDM_E_all_subjects_per_condition_per_subject[:,:,p],
                        overall_acc_GDM_S_all_subjects_per_condition_per_subject[:,:,p],
                        order=None,location=save_per_subject_individual, subject=subjects[p])

        plotting_scores(experimentManager, all_seen_f1_GDM_E_all_subjects_per_condition_per_subject[:, :, p],
                        all_seen_f1_GDM_S_all_subjects_per_condition_per_subject[:, :, p],
                        overall_f1_GDM_E_all_subjects_per_condition_per_subject[:, :, p],
                        overall_f1_GDM_S_all_subjects_per_condition_per_subject[:, :, p],
                        order=None, location=save_per_subject_individual, subject=subjects[p], f1=True)



def plot_violin(data, location=None, incremental=True, episodic=True, type="", individual=False, condition="",
                kruskal_stats=None, anderson_stats=None):


    # if type.startswith("Per_Order"):
    for c in range(data.shape[2]):
        plt.figure(figsize=(9.6, 7.2), dpi=150)
        plt.ylim((-0.1, 1.1))
        if incremental:
            title = "Incremental Learning " + type
            plt.title(title)
        else:
            title = "Overall Accuracy " + type
            plt.title(title)

        if individual:
            plt.xlabel("Orders")
        else:
            plt.xlabel("Conditions")

        plt.ylabel("Accuracy")
        if individual:
            plt.xticks(np.arange(0, len(data)), tuple([str(i+1) for i in range(len(data))]))
            dataframe = pd.DataFrame(data=data[:,:,c].T, columns=tuple([str(i+1) for i in range(len(data))]))
        else:
            # if episodic:
            plt.xticks(np.arange(0, 3), ("GDM", "GDM + Replay", "GDM + Imagination"))
            dataframe = pd.DataFrame(data=data[:,:,c].T, columns=["GDM", "GDM + Replay", "GDM + Imagination"])
            # else:
            #     plt.xticks(np.arange(0, 3), ("GDM-S", "GDM-S + Replay", "GDM-S + Imagination"))
            #     dataframe = pd.DataFrame(data=data.T, columns=["GDM-S", "GDM-S + Replay", "GDM-S + Imagination"])
        # plt.tight_layout()
        sns.violinplot(data=dataframe,meanline=True, showmeans=True, cut=0)
        if type.startswith("Per_Order"):
            if incremental:
                if individual:
                    plt.savefig(location + "/" + condition + "_Order_wise_distributions_incremental_violin" +
                                    "_after_class_" + str(c) + ".png",
                                    bbox_inches='tight')
                else:
                    if episodic:
                        plt.savefig(location + "/Order_wise_distributions_incremental_episodic_violin" +
                                    "_after_class_" + str(c) + ".png",
                                bbox_inches='tight')
                    else:
                        plt.savefig(location + "/Order_wise_distributions_incremental_semantic_violin" +
                                    "_after_class_" + str(c) + ".png",
                                    bbox_inches='tight')
            else:
                if individual:
                    plt.savefig(location + "/" + condition + "_Order_wise_distributions_overall_violin" +
                                    "_after_class_" + str(c) + ".png",
                                    bbox_inches='tight')
                else:
                    if episodic:
                        plt.savefig(location + "/Order_wise_distributions_overall_episodic_violin" +
                                    "_after_class_" + str(c) + ".png",
                                bbox_inches='tight')
                    else:
                        plt.savefig(location + "/Order_wise_distributions_overall_semantic_violin" +
                                    "_after_class_" + str(c) + ".png",
                                    bbox_inches='tight')
        else:
            if incremental:
                if individual:
                    plt.savefig(location + "/" + condition + "_subject_wise_distributions_incremental_violin" +
                                    "_after_class_" + str(c) + ".png",
                                bbox_inches='tight')
                else:
                    if episodic:
                        plt.savefig(location + "/subject_wise_distributions_incremental_episodic_violin" +
                                    "_after_class_" + str(c) + ".png",
                                bbox_inches='tight')
                    else:
                        plt.savefig(location + "/subject_wise_distributions_incremental semantic_violin" +
                                    "_after_class_" + str(c) + ".png",
                                    bbox_inches='tight')
            else:
                if individual:
                    plt.savefig(location + "/" + condition + "_subject_wise_distributions_overall_violin" +
                                    "_after_class_" + str(c) + ".png",
                                    bbox_inches='tight')
                else:
                    if episodic:
                        plt.savefig(location + "/subject_wise_distributions_overall_episodic_violin" +
                                    "_after_class_" + str(c) + ".png",
                                bbox_inches='tight')
                    else:
                        plt.savefig(location + "/subject_wise_distributions_overall semantic_violin" +
                                    "_after_class_" + str(c) + ".png",
                                    bbox_inches='tight')

        plt.gcf().clear()
    plot_box(data=data, location=location, incremental=incremental, episodic=episodic, type=type,
                 individual=individual,
                 condition=condition, kruskal_stats=kruskal_stats, anderson_stats=anderson_stats)

def plot_box(data, location=None, incremental=True, episodic=True, type="", individual=False, condition="",
                kruskal_stats=None, anderson_stats=None):


    for c in range(data.shape[2]):

        plt.figure(figsize=(9.6, 7.2), dpi=150)
        plt.ylim((-0.1, 1.1))
        if incremental:
            title = "Incremental Learning " + type
            plt.title(title)
        else:
            title = "Overall Accuracy " + type
            plt.title(title)

        if individual:
            plt.xlabel("Number of Seen Classes")
        else:
            plt.xlabel("Conditions")

        plt.ylabel("Accuracy")
        if individual:
            plt.xticks(np.arange(0, len(data)), tuple([str(i+1) for i in range(len(data))]))
            dataframe = pd.DataFrame(data=data[:,:,c].T, columns=tuple([str(i+1) for i in range(len(data))]))
        else:
            plt.xticks(np.arange(0, 3), ("GDM", "GDM + Replay", "GDM + Imagination"))
            dataframe = pd.DataFrame(data=data[:,:,c].T, columns=["GDM", "GDM + Replay", "GDM + Imagination"])

        sns.boxplot(data=dataframe,meanline=True, showmeans=True)

        if type.startswith("Per_Order"):
            if incremental:
                if individual:
                    plt.savefig(location + "/" + condition + "_Order_wise_distributions_incremental_box" +
                                    "_after_class_" + str(c) + ".png",
                                    bbox_inches='tight')
                else:
                    if episodic:
                        plt.savefig(location + "/Order_wise_distributions_incremental_episodic_box" +
                                    "_after_class_" + str(c) + ".png",
                                bbox_inches='tight')
                    else:
                        plt.savefig(location + "/Order_wise_distributions_incremental_semantic_box" +
                                    "_after_class_" + str(c) + ".png",
                                    bbox_inches='tight')
            else:
                if individual:
                    plt.savefig(location + "/" + condition + "_Order_wise_distributions_overall_box" +
                                    "_after_class_" + str(c) + ".png",
                                    bbox_inches='tight')
                else:
                    if episodic:
                        plt.savefig(location + "/Order_wise_distributions_overall_episodic_box" +
                                    "_after_class_" + str(c) + ".png",
                                bbox_inches='tight')
                    else:
                        plt.savefig(location + "/Order_wise_distributions_overall_semantic_box" +
                                    "_after_class_" + str(c) + ".png",
                                    bbox_inches='tight')
        else:
            if incremental:
                if individual:
                    plt.savefig(location + "/" + condition + "_subject_wise_distributions_incremental_box" +
                                    "_after_class_" + str(c) + ".png",
                                bbox_inches='tight')
                else:
                    if episodic:
                        plt.savefig(location + "/subject_wise_distributions_incremental_episodic_box" +
                                    "_after_class_" + str(c) + ".png",
                                bbox_inches='tight')
                    else:
                        plt.savefig(location + "/subject_wise_distributions_incremental semantic_box" +
                                    "_after_class_" + str(c) + ".png",
                                    bbox_inches='tight')
            else:
                if individual:
                    plt.savefig(location + "/" + condition + "_subject_wise_distributions_overall_box" +
                                    "_after_class_" + str(c) + ".png",
                                    bbox_inches='tight')
                else:
                    if episodic:
                        plt.savefig(location + "/subject_wise_distributions_overall_episodic_box" +
                                    "_after_class_" + str(c) + ".png",
                                bbox_inches='tight')
                    else:
                        plt.savefig(location + "/subject_wise_distributions_overall semantic_box" +
                                    "_after_class_" + str(c) + ".png",
                                    bbox_inches='tight')

        plt.gcf().clear()

def plotting_scores(experimentManager, all_seen_acc_GDM_E_all_subjects_per_condition,
                        all_seen_acc_GDM_S_all_subjects_per_condition, overall_acc_GDM_E_all_subjects_per_condition,
                        overall_acc_GDM_S_all_subjects_per_condition, order, location=None, subject=None, f1=False):
    def compute_mean_ci(data, ci=0.95):
        mu = mean(data, axis=0)
        std_err = sem(data)
        h = std_err * t.ppf((1 + ci) / 2, len(data) - 1)
        return mu, std_err, h
    labels_e = ['MLP Baseline', 'GDM-E', 'GDM-E + Replay', 'GDM-E + Imagination']
    labels_s = ['MLP Baseline', 'GDM-S', 'GDM-S + Replay', 'GDM-S + Imagination']
    colors = ["green", "red", "blue", "black"]

    if len(all_seen_acc_GDM_E_all_subjects_per_condition) >3: # If using MLP baseline
        MLP_all_seen_acc_GDM_E_plot = pd.DataFrame(all_seen_acc_GDM_E_all_subjects_per_condition[0])
        MLP_all_seen_acc_GDM_S_plot = pd.DataFrame(all_seen_acc_GDM_S_all_subjects_per_condition[0])

        GDM_all_seen_acc_GDM_E_plot = pd.DataFrame(all_seen_acc_GDM_E_all_subjects_per_condition[1])
        GDM_all_seen_acc_GDM_S_plot = pd.DataFrame(all_seen_acc_GDM_S_all_subjects_per_condition[1])
        GDM_replay_all_seen_acc_GDM_E_plot = pd.DataFrame(all_seen_acc_GDM_E_all_subjects_per_condition[2])
        GDM_replay_all_seen_acc_GDM_S_plot = pd.DataFrame(all_seen_acc_GDM_S_all_subjects_per_condition[2])
        GDM_imagine__all_seen_acc_GDM_E_plot = pd.DataFrame(all_seen_acc_GDM_E_all_subjects_per_condition[3])
        GDM_imagine__all_seen_acc_GDM_S_plot = pd.DataFrame(all_seen_acc_GDM_S_all_subjects_per_condition[3])

        MLP_overall_acc_GDM_E_plot = pd.DataFrame(overall_acc_GDM_E_all_subjects_per_condition[0])
        MLP_overall_acc_GDM_S_plot = pd.DataFrame(overall_acc_GDM_S_all_subjects_per_condition[0])

        GDM_overall_acc_GDM_E_plot = pd.DataFrame(overall_acc_GDM_E_all_subjects_per_condition[1])
        GDM_overall_acc_GDM_S_plot = pd.DataFrame(overall_acc_GDM_S_all_subjects_per_condition[1])
        GDM_replay_overall_acc_GDM_E_plot = pd.DataFrame(overall_acc_GDM_E_all_subjects_per_condition[2])
        GDM_replay_overall_acc_GDM_S_plot = pd.DataFrame(overall_acc_GDM_S_all_subjects_per_condition[2])
        GDM_imagine_overall_acc_GDM_E_plot = pd.DataFrame(overall_acc_GDM_E_all_subjects_per_condition[3])
        GDM_imagine_overall_acc_GDM_S_plot = pd.DataFrame(overall_acc_GDM_S_all_subjects_per_condition[3])
    else:
        GDM_all_seen_acc_GDM_E_plot = pd.DataFrame(all_seen_acc_GDM_E_all_subjects_per_condition[0])
        GDM_all_seen_acc_GDM_S_plot = pd.DataFrame(all_seen_acc_GDM_S_all_subjects_per_condition[0])
        GDM_replay_all_seen_acc_GDM_E_plot = pd.DataFrame(all_seen_acc_GDM_E_all_subjects_per_condition[1])
        GDM_replay_all_seen_acc_GDM_S_plot = pd.DataFrame(all_seen_acc_GDM_S_all_subjects_per_condition[1])
        GDM_imagine__all_seen_acc_GDM_E_plot = pd.DataFrame(all_seen_acc_GDM_E_all_subjects_per_condition[2])
        GDM_imagine__all_seen_acc_GDM_S_plot = pd.DataFrame(all_seen_acc_GDM_S_all_subjects_per_condition[2])

        GDM_overall_acc_GDM_E_plot = pd.DataFrame(overall_acc_GDM_E_all_subjects_per_condition[0])
        GDM_overall_acc_GDM_S_plot = pd.DataFrame(overall_acc_GDM_S_all_subjects_per_condition[0])
        GDM_replay_overall_acc_GDM_E_plot = pd.DataFrame(overall_acc_GDM_E_all_subjects_per_condition[1])
        GDM_replay_overall_acc_GDM_S_plot = pd.DataFrame(overall_acc_GDM_S_all_subjects_per_condition[1])
        GDM_imagine_overall_acc_GDM_E_plot = pd.DataFrame(overall_acc_GDM_E_all_subjects_per_condition[2])
        GDM_imagine_overall_acc_GDM_S_plot = pd.DataFrame(overall_acc_GDM_S_all_subjects_per_condition[2])

    if f1:
        results = ["Incremental Learning", "Overall F1 Score "]
    else:
        results = ["Incremental Learning", "Overall Accuracy "]
    plots = ["Episodic Memory", "Semantic Memory"]

    for r in results:
        for p in plots:
            plt.figure(figsize=(9.6, 7.2), dpi=150)
            plt.ylim((-0.1, 1.1))
            plt.title(r)
            plt.xlabel("Number of Classes Encountered")
            if f1:
                plt.ylabel("F1 Score")
            else:
                plt.ylabel("Accuracy")
            plt.xticks(np.arange(0, 6), ("1", "2", "3", "4", "5", "6"))
            plt.tight_layout()
            if r.startswith("Incremental"):
                if p.startswith("Episodic"):
                    if len(all_seen_acc_GDM_E_all_subjects_per_condition) >3:
                        sns.tsplot(data=MLP_all_seen_acc_GDM_E_plot.values, ci=95, color=colors[0], marker="x")
                        if f1:
                            experimentManager.logManager.write("MLP_all_seen_f1_GDM_E Mean, Std. Err & CI:")
                        else:
                            experimentManager.logManager.write("MLP_all_seen_acc_GDM_E Mean, Std. Err & CI:")
                        experimentManager.logManager.write(compute_mean_ci(MLP_all_seen_acc_GDM_E_plot.values))

                    sns.tsplot(data=GDM_all_seen_acc_GDM_E_plot.values, ci=95, color=colors[1], marker="o")
                    if f1:
                        experimentManager.logManager.write("GDM_all_seen_f1_GDM_E Mean, Std. Err & CI:")
                    else:
                        experimentManager.logManager.write("GDM_all_seen_acc_GDM_E Mean, Std. Err & CI:")
                    experimentManager.logManager.write(compute_mean_ci(GDM_all_seen_acc_GDM_E_plot.values))

                    sns.tsplot(data=GDM_replay_all_seen_acc_GDM_E_plot.values, ci=95, color=colors[2], marker="^")
                    if f1:
                        experimentManager.logManager.write("GDM_replay_all_seen_f1_GDM_E Mean, Std. Err & CI:")
                    else:
                        experimentManager.logManager.write("GDM_replay_all_seen_acc_GDM_E Mean, Std. Err & CI:")
                    experimentManager.logManager.write(compute_mean_ci(GDM_replay_all_seen_acc_GDM_E_plot.values))

                    sns.tsplot(data=GDM_imagine__all_seen_acc_GDM_E_plot.values, ci=95, color=colors[3], marker="s")
                    if f1:
                        experimentManager.logManager.write("GDM_imagine__all_seen_f1_GDM_E Mean, Std. Err & CI:")
                    else:
                        experimentManager.logManager.write("GDM_imagine__all_seen_acc_GDM_E Mean, Std. Err & CI:")
                    experimentManager.logManager.write(compute_mean_ci(GDM_imagine__all_seen_acc_GDM_E_plot.values))
                    plt.legend(labels_e, loc='best')

                else:
                    if len(all_seen_acc_GDM_E_all_subjects_per_condition) >3:

                        sns.tsplot(data=MLP_all_seen_acc_GDM_S_plot.values, ci=95, color=colors[0], marker="x")
                        if f1:
                            experimentManager.logManager.write("MLP_all_seen_f1_GDM_S Mean, Std. Err & CI:")
                        else:
                            experimentManager.logManager.write("MLP_all_seen_acc_GDM_S Mean, Std. Err & CI:")
                        experimentManager.logManager.write(compute_mean_ci(MLP_all_seen_acc_GDM_S_plot.values))

                    sns.tsplot(data=GDM_all_seen_acc_GDM_S_plot.values, ci=95, color=colors[1], marker="o")
                    if f1:
                        experimentManager.logManager.write("GDM_all_seen_f1_GDM_S Mean, Std. Err & CI:")
                    else:
                        experimentManager.logManager.write("GDM_all_seen_acc_GDM_S Mean, Std. Err & CI:")
                    experimentManager.logManager.write(compute_mean_ci(GDM_all_seen_acc_GDM_S_plot.values))

                    sns.tsplot(data=GDM_replay_all_seen_acc_GDM_S_plot.values, ci=95, color=colors[2], marker="^")
                    if f1:
                        experimentManager.logManager.write("GDM_replay_all_seen_f1_GDM_S Mean, Std. Err & CI:")
                    else:
                        experimentManager.logManager.write("GDM_replay_all_seen_acc_GDM_S Mean, Std. Err & CI:")
                    experimentManager.logManager.write(compute_mean_ci(GDM_replay_all_seen_acc_GDM_S_plot.values))

                    sns.tsplot(data=GDM_imagine__all_seen_acc_GDM_S_plot.values, ci=95, color=colors[3], marker="s")
                    if f1:
                        experimentManager.logManager.write("GDM_imagine__all_seen_f1_GDM_S Mean, Std. Err & CI:")
                    else:
                        experimentManager.logManager.write("GDM_imagine__all_seen_acc_GDM_S Mean, Std. Err & CI:")
                    experimentManager.logManager.write(compute_mean_ci(GDM_imagine__all_seen_acc_GDM_S_plot.values))
                    plt.legend(labels_s, loc='best')

            else:
                if p.startswith("Episodic"):
                    if len(all_seen_acc_GDM_E_all_subjects_per_condition) >3:
                        sns.tsplot(data=MLP_overall_acc_GDM_E_plot.values, ci=95, color=colors[0], marker="x")
                        if f1:
                            experimentManager.logManager.write("MLP_overall_f1_GDM_E Mean, Std. Err & CI:")
                        else:
                            experimentManager.logManager.write("MLP_overall_acc_GDM_E Mean, Std. Err & CI:")
                        experimentManager.logManager.write(compute_mean_ci(MLP_overall_acc_GDM_E_plot.values))

                    sns.tsplot(data=GDM_overall_acc_GDM_E_plot.values, ci=95, color=colors[1], marker="o")
                    if f1:
                        experimentManager.logManager.write("GDM_overall_f1_GDM_E Mean, Std. Err & CI:")
                    else:
                        experimentManager.logManager.write("GDM_overall_acc_GDM_E Mean, Std. Err & CI:")
                    experimentManager.logManager.write(compute_mean_ci(GDM_overall_acc_GDM_E_plot.values))

                    sns.tsplot(data=GDM_replay_overall_acc_GDM_E_plot.values, ci=95, color=colors[2], marker="^")
                    if f1:
                        experimentManager.logManager.write("GDM_replay_overall_f1_GDM_E Mean, Std. Err & CI:")
                    else:
                        experimentManager.logManager.write("GDM_replay_overall_acc_GDM_E Mean, Std. Err & CI:")
                    experimentManager.logManager.write(compute_mean_ci(GDM_replay_overall_acc_GDM_E_plot.values))

                    sns.tsplot(data=GDM_imagine_overall_acc_GDM_E_plot.values, ci=95, color=colors[3], marker="s")
                    if f1:
                        experimentManager.logManager.write("GDM_imagine_overall_f1_GDM_E Mean, Std. Err & CI:")
                    else:
                        experimentManager.logManager.write("GDM_imagine_overall_acc_GDM_E Mean, Std. Err & CI:")
                    experimentManager.logManager.write(compute_mean_ci(GDM_imagine_overall_acc_GDM_E_plot.values))
                    plt.legend(labels_e, loc='best')

                else:
                    if len(all_seen_acc_GDM_E_all_subjects_per_condition) >3:
                        sns.tsplot(data=MLP_overall_acc_GDM_S_plot.values, ci=95, color=colors[0], marker="x")
                        if f1:
                            experimentManager.logManager.write("MLP_overall_f1_GDM_S Mean, Std. Err & CI:")
                        else:
                            experimentManager.logManager.write("MLP_overall_acc_GDM_S Mean, Std. Err & CI:")
                        experimentManager.logManager.write(compute_mean_ci(MLP_overall_acc_GDM_S_plot.values))

                    sns.tsplot(data=GDM_overall_acc_GDM_S_plot.values, ci=95, color=colors[1], marker="o")
                    if f1:
                        experimentManager.logManager.write("GDM_overall_f1_GDM_S Mean, Std. Err & CI:")
                    else:
                        experimentManager.logManager.write("GDM_overall_acc_GDM_S Mean, Std. Err & CI:")
                    experimentManager.logManager.write(compute_mean_ci(GDM_overall_acc_GDM_S_plot.values))

                    sns.tsplot(data=GDM_replay_overall_acc_GDM_S_plot.values, ci=95, color=colors[2], marker="^")
                    if f1:
                        experimentManager.logManager.write("GDM_replay_overall_f1_GDM_S Mean, Std. Err & CI:")
                    else:
                        experimentManager.logManager.write("GDM_replay_overall_acc_GDM_S Mean, Std. Err & CI:")
                    experimentManager.logManager.write(compute_mean_ci(GDM_replay_overall_acc_GDM_S_plot.values))

                    sns.tsplot(data=GDM_imagine_overall_acc_GDM_S_plot.values, ci=95, color=colors[3], marker="s")
                    if f1:
                        experimentManager.logManager.write("GDM_imagine_overall_f1_GDM_S_plot Mean, Std. Err & CI:")
                    else:
                        experimentManager.logManager.write("GDM_imagine_overall_acc_GDM_S_plot Mean, Std. Err & CI:")
                    experimentManager.logManager.write(compute_mean_ci(GDM_imagine_overall_acc_GDM_S_plot.values))
                    plt.legend(labels_s, loc='best')

            if location is None:
                if f1:
                    plt.savefig(experimentManager.plotsDirectory + "/" + str(order) + "_" + r + "_" + p + "_f1.png",
                        bbox_inches='tight')
                else:
                    plt.savefig(experimentManager.plotsDirectory + "/" + str(order) + "_" + r + "_" + p + ".png",
                                bbox_inches='tight')
            else:
                if subject is None:
                    if f1:
                        plt.savefig(location + "/" + str(order) + "_" + r + "_" + p + "_f1.png",
                            bbox_inches='tight')
                    else:
                        plt.savefig(location + "/" + str(order) + "_" + r + "_" + p + ".png",
                                    bbox_inches='tight')
                else:
                    if f1:
                        plt.savefig(location + "/" + str(subject) + "_" + r + "_" + p + "_f1.png",
                                    bbox_inches='tight')
                    else:
                        plt.savefig(location + "/" + str(subject) + "_" + r + "_" + p + ".png",
                                    bbox_inches='tight')

            plt.gcf().clear()

def classifier(experimentManager, order, args, labelsize):
    from keras.layers import Dense, BatchNormalization
    from keras.models import Model, Input
    from keras.initializers import Constant, RandomNormal

    experimentManager.logManager.newLogSession("Implementing Classifier C Model: " + str(order) + "_C")

    """ ******************************************************************************"""
    """ Classifier C Model """
    """ ******************************************************************************"""

    c_input = Input(shape=[args.zshape], name="c_Input")

    c_dense_1 = Dense(64, activation='relu', use_bias=True,bias_initializer=Constant(0.0),
                      kernel_initializer=RandomNormal(stddev=0.02), name="c_dense_1")(c_input)
    c_batch_norm_1 = BatchNormalization(scale=False, name='z_batch_norm_1')(c_dense_1)

    c_dense_2 = Dense(32, activation='relu', use_bias=True,bias_initializer=Constant(0.0),
                      kernel_initializer=RandomNormal(stddev=0.025), name="c_dense_2")(c_batch_norm_1)
    c_batch_norm_2 = BatchNormalization(scale=False, name='c_batch_norm_2')(c_dense_2)

    c_dense_3 = Dense(16, activation='relu', use_bias=True,bias_initializer=Constant(0.0),
                      kernel_initializer=RandomNormal(stddev=0.025), name="z_dense_3")(c_batch_norm_2)
    c_out = Dense(labelsize, activation='softmax', use_bias=True,bias_initializer=Constant(0.0),
                  kernel_initializer=RandomNormal(stddev=0.025),name="c_out")(c_dense_3)

    classifier_model = Model(inputs=c_input, outputs=c_out, name='Classifier')

    classifier_model.summary()

    experimentManager.logManager.write(
        "--- Plotting and saving the C model at: " + str(experimentManager.plotManager.plotsDirectory) +
        "/" + str(order) + "_C_plot.png")

    experimentManager.plotManager.creatModelPlot(classifier_model, str(order) + "_C")

    experimentManager.logManager.endLogSession()
    return classifier_model

def test_mlp(args, data, labels, CAAE_model, order, first=False, classifier_model=None, train=False):
    from keras.utils import np_utils
    from keras.optimizers import Adam
    from keras.models import load_model
    from KEF.CustomObjects.metrics import fmeasure

    labels_train = np_utils.to_categorical(y=labels, num_classes=args.classes)
    if first:
        classifier_model = classifier(experimentManager=experimentManager, order=order, args=args,
                                      labelsize=args.classes)
        optimizer = Adam(lr=args.learningrate, beta_1=0.5, beta_2=0.999, decay=0.)
        classifier_model.compile(loss="categorical_crossentropy", optimizer=optimizer,
                                 metrics=['categorical_accuracy', fmeasure])
    if train:
        history_callback = classifier_model.fit(x=data, y=labels_train,
                                                batch_size=args.batch_size,
                                                epochs=20)
    evaluation = classifier_model.evaluate(data, labels_train, batch_size=args.batch_size)

    return classifier_model, (evaluation[1], evaluation[1]), (evaluation[2],evaluation[2])


def run(experimentManager, args, preProcessingProperties, datasetTrainFolder, labelCSV_train):

    dataLoader = DataLoader_CAAE_Categorical.DataLoader_CAAE_Categorical(experimentManager.logManager,
                                                                         preProcessingProperties)
    # Populating a list with all the subject IDs. For eg. 1-24
    if args.dataset.startswith("RAV"):
        subjects = [i.astype(str).zfill(2) for i in np.arange(1,args.subjects + 1)]
        class_dictionary = {0: "Angry", 1: "Fear", 2: "Happy", 3: "Sad", 4: "Surprise", 5: "Neutral"}

    elif args.dataset.startswith("MMI"):
        datasetLabels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        true_classes = sort_nicely(datasetLabels)
        true_classes.pop(1)
        subjects = sort_nicely(os.listdir(datasetTrainFolder))
        args.subjects = len(subjects)
        class_dictionary = {0: "Angry", 1: "Fear", 2: "Happy", 3: "Sad", 4: "Surprise", 5: "Neutral"}

    elif args.dataset.startswith("BAUM"):
        datasetLabels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
        true_classes = sort_nicely(datasetLabels)
        true_classes = [e.lower() for e in true_classes]
        true_classes.pop(1)
        all_subjects = sort_nicely(os.listdir(datasetTrainFolder))
        subjects = []
        for subject in all_subjects:
            classes = sort_nicely(os.listdir(datasetTrainFolder + "/" + subject))
            classes = [c.lower() for c in classes]
            if all(elem in classes for elem in true_classes):
                subjects.append(subject)
        args.subjects = len(subjects)
        class_dictionary = {0: "Anger", 1: "Fear", 2: "Happiness", 3: "Sadness", 4: "Surprise", 5: "Neutral"}

    # Loading the CAAE-based Imagination Module
    CAAE_LoadPath = "KEF/Models/Trained_Model"

    CAAE = loadModel(experimentManager, dataLoader, args, CAAE_LoadPath, preProcessingProperties)

    # MLP included in the comparison as condition 0.
    conditions = {0: "MLP", 1: "GDM", 2: "GDM + Replay", 3: "GDM + Imagination"}

    # Variables accumulating the results across all subjects, for all conditions, and all orders
    all_seen_acc_GDM_E_all_per_order = []
    all_seen_acc_GDM_S_all_per_order = []
    overall_acc_GDM_E_all_per_order = []
    overall_acc_GDM_S_all_per_order = []

    all_seen_f1_GDM_E_all_per_order = []
    all_seen_f1_GDM_S_all_per_order = []
    overall_f1_GDM_E_all_per_order = []
    overall_f1_GDM_S_all_per_order = []

    # Populating all permutations of class orders.
    class_orders = list(itertools.permutations(np.arange(0, args.classes)))
    # Fixing Class Orders
    # class_dictionary = {0: "Anger", 1: "Disgust", 2: "Fear", 3: "Happiness", 4: "Sadness", 5: "Surprise", 6: "Neutral"}
    # class_orders = [(0, 4, 3, 5, 1, 2), (1, 3, 0, 5, 2, 4), (2, 3, 4, 0, 5, 1), (3, 1, 2, 4, 5, 0), (4, 2, 0, 3, 5, 1),
    #                 (5, 1, 4, 0, 2, 3)]

    labelled_orders = []
    # Saving all populated Class orders to CSV file
    for order in class_orders:
        labels = [class_dictionary.get(i, i) for i in order]
        labelled_orders.append(dict(list(enumerate(labels))))
    with open(experimentManager.baseDirectory + "/" + experimentManager.experimentName + '/class_orders.csv', 'w') \
            as output_file:
        dict_writer = csv.DictWriter(output_file, labelled_orders[0])
        dict_writer.writeheader()
        dict_writer.writerows(labelled_orders)


    # Running the Experiment Loop for all class_orders.
    for idx, order in enumerate(class_orders):
        save_path = experimentManager.outputsDirectory + "/" + str(idx)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Get Class Labels for the order
        effective_loaded_labels = [class_dictionary.get(i, i) for i in order]
        order_label_dictionary = dict(list(enumerate(effective_loaded_labels)))
        experimentManager.logManager.write("Class order followed: " + str(order_label_dictionary))

        # Variables accumulating the results across all subjects and all conditions for a given order
        novel_acc_GDM_E_all_subjects_per_condition = []
        novel_acc_GDM_S_all_subjects_per_condition = []
        all_seen_acc_GDM_E_all_subjects_per_condition = []
        all_seen_acc_GDM_S_all_subjects_per_condition = []
        overall_acc_GDM_E_all_subjects_per_condition = []
        overall_acc_GDM_S_all_subjects_per_condition = []

        novel_f1_GDM_E_all_subjects_per_condition = []
        novel_f1_GDM_S_all_subjects_per_condition = []
        all_seen_f1_GDM_E_all_subjects_per_condition = []
        all_seen_f1_GDM_S_all_subjects_per_condition = []
        overall_f1_GDM_E_all_subjects_per_condition = []
        overall_f1_GDM_S_all_subjects_per_condition = []

        # Running the Experiment Loop for all conditions for a given class order.
        for cond in conditions:
            if cond == 0:
                # Running MLP Condition
                replay = False
                imagine = False
                mlp_classifier = True
            elif cond == 1:
                # Running GDM condition.
                replay = False
                imagine = False
                mlp_classifier = False
            elif cond == 2:
                # Running GDM + Replay condition.
                replay = True
                imagine = False
                mlp_classifier = False
            else:
                # Running GDM + Imagination condition.
                replay = False
                imagine = True
                mlp_classifier = False

            experimentManager.logManager.write("Condition : " + str(conditions[cond]))

            # Variables accumulating the results across all subjects for a given condition and a given order
            # MLP results included as index 0 for each variable.
            novel_acc_GDM_E_all_subjects = []
            novel_acc_GDM_S_all_subjects = []

            all_seen_acc_GDM_E_all_subjects = []
            all_seen_acc_GDM_S_all_subjects = []

            overall_acc_GDM_E_all_subjects = []
            overall_acc_GDM_S_all_subjects = []

            novel_f1_GDM_E_all_subjects = []
            novel_f1_GDM_S_all_subjects = []

            all_seen_f1_GDM_E_all_subjects = []
            all_seen_f1_GDM_S_all_subjects = []

            overall_f1_GDM_E_all_subjects = []
            overall_f1_GDM_S_all_subjects = []

            for subject in subjects:

                experimentManager.logManager.write("Running for Subject: " + str(subject))

                # Loading All data for a given subject for Testing for Ovearall Accuracy Task.
                dataLoader_test_all = loadData(dataLoader, datasetTrainFolder, labelCSV_train, dataSet=args.dataset,
                                           dataType="Test", subjectID=subject, order=effective_loaded_labels)


                # List of all previously seen classes.
                emotions = []
                # Initialising Episodic and Semantic Memory models from None.
                trained_GWRs = (None, None)
                # If running for the first time for a subject, load testing data.
                new = True
                first_classify = True
                # Variables accumulating the results for a given subject, for a given condition and a given order
                # MLP results included as index 0 for each variable.

                novel_acc_GDM_E = []
                all_seen_acc_GDM_E = []
                overall_acc_GDM_E = []
                novel_acc_GDM_S = []
                all_seen_acc_GDM_S = []
                overall_acc_GDM_S = []

                novel_f1_GDM_E = []
                all_seen_f1_GDM_E = []
                overall_f1_GDM_E = []
                novel_f1_GDM_S = []
                all_seen_f1_GDM_S = []
                overall_f1_GDM_S = []

                # Running for all classes
                for emo in range(1, args.classes + 1):

                    # Appending current class to all seen classes
                    emotions.append(emo)
                    # Dataloader for loading current class for training
                    dataLoader_novel = loadData(dataLoader, datasetTrainFolder, labelCSV_train, dataSet=args.dataset,
                                                dataType="Train", emo=[emo], subjectID=subject,
                                                order=effective_loaded_labels)
                    novel_data, novel_labels = CAAE.generate_data(dataPoint=dataLoader_novel.dataTrain)

                    # Dataloader for loading all seen classes for testing
                    dataLoader_all_seen = loadData(dataLoader, datasetTrainFolder, labelCSV_train, dataSet=args.dataset,
                                                   dataType="Valid", emo=emotions,
                                                   subjectID=subject, order=effective_loaded_labels)
                    all_seen_data, all_seen_labels = CAAE.generate_data(dataPoint=dataLoader_all_seen.dataValidation)

                    # If running for the first time, load all data from all classes for Overall Accuracy Testing.
                    if new:
                        all_data, all_labels = CAAE.generate_data(dataPoint=dataLoader_test_all.dataTest)
                        new = False

                    experimentManager.logManager.write("Training for: " + str(effective_loaded_labels[emo - 1]))

                    """##############################################################################################"""
                    """ ##############################    Training with Novel Data   ################################"""
                    """##############################################################################################"""
                    trained_GWRs, accs, f1s = train_GDM(data=novel_data, labels=novel_labels, CAAE_model=CAAE,
                                                        trained_GWRs=trained_GWRs, labelsize=len(class_dictionary),
                                                        train_replay=replay, train_imagine=imagine,
                                                        order_label_dictionary=order_label_dictionary, args=args,
                                                        mlp_classifier=mlp_classifier, first_classify=first_classify)
                    if first_classify:
                        first_classify = False
                    # Logging Episodic and Semantic accuracies for current emotional class for the subject
                    novel_acc_GDM_E.append(accs[0])
                    novel_f1_GDM_E.append(f1s[0])
                    novel_acc_GDM_S.append(accs[1])
                    novel_f1_GDM_S.append(f1s[1])

                    """##############################################################################################"""
                    """ ############################    Testing with All Seen Data   ################################"""
                    """##############################################################################################"""
                    experimentManager.logManager.write("Testing for All Seen")
                    all_seen_accs, all_seen_f1s = test_GDM(trained_GWRs=trained_GWRs, test_data=all_seen_data,
                                                           test_labels=all_seen_labels, labelsize=len(class_dictionary),
                                                           type="All Seen", args=args, mlp_classifier=mlp_classifier,
                                                           CAAE_model=CAAE,
                                                           order_label_dictionary=order_label_dictionary,
                                                           classifier_model=trained_GWRs[0])
                    # Logging Episodic and Semantic accuracies for all emotional classes seen for the subject
                    all_seen_acc_GDM_E.append(all_seen_accs[0])
                    all_seen_f1_GDM_E.append(all_seen_f1s[0])
                    all_seen_acc_GDM_S.append(all_seen_accs[1])
                    all_seen_f1_GDM_S.append(all_seen_f1s[1])

                    """##############################################################################################"""
                    """ ###############################    Testing with All Data   ##################################"""
                    """##############################################################################################"""
                    experimentManager.logManager.write("Testing for all data")
                    overall_accs, overall_f1s = test_GDM(trained_GWRs=trained_GWRs, test_data=all_data,
                                                         test_labels=all_labels, labelsize=len(class_dictionary),
                                                         type="Overall", args=args, mlp_classifier=mlp_classifier,
                                                         CAAE_model=CAAE, order_label_dictionary=order_label_dictionary,
                                                         first=first_classify, classifier_model=trained_GWRs[0])
                    # Logging Episodic and Semantic accuracies for Overall accuracy task for the subject
                    overall_acc_GDM_E.append(overall_accs[0])
                    overall_f1_GDM_E.append(overall_f1s[0])
                    overall_acc_GDM_S.append(overall_accs[1])
                    overall_f1_GDM_S.append(overall_f1s[1])

                """##################################################################################################"""
                """#############################    Printing scores for each subject  ###############################"""
                """##################################################################################################"""

                experimentManager.logManager.write("Order :  " + str(order_label_dictionary))
                experimentManager.logManager.write("Condition : " + str(conditions[cond]))
                """##################################    Scores on Novel Data  ######################################"""
                experimentManager.logManager.write(
                    "Accuracy on Novel Data for subject:" + subject + " for Episodic Memory:")
                experimentManager.logManager.write(novel_acc_GDM_E)
                # Appending to results for all subjects for the given condition and given order
                novel_acc_GDM_E_all_subjects.append(novel_acc_GDM_E)
                experimentManager.logManager.write(
                    "Accuracy on Novel Data for subject:" + subject + " for Semantic Memory:")
                experimentManager.logManager.write(novel_acc_GDM_S)
                # Appending to results for all subjects for the given condition and given order
                novel_acc_GDM_S_all_subjects.append(novel_acc_GDM_S)
                """##################################    F1 Scores on Novel Data  ###################################"""

                experimentManager.logManager.write(
                    "F1 Score on Novel Data for subject:" + subject + " for Episodic Memory:")
                experimentManager.logManager.write(novel_f1_GDM_E)
                # Appending to results for all subjects for the given condition and given order
                novel_f1_GDM_E_all_subjects.append(novel_f1_GDM_E)
                experimentManager.logManager.write(
                    "F1 Score on Novel Data for subject:" + subject + " for Semantic Memory:")
                experimentManager.logManager.write(novel_f1_GDM_S)
                # Appending to results for all subjects for the given condition and given order
                novel_f1_GDM_S_all_subjects.append(novel_f1_GDM_S)

                """##################################    Scores on all seen Data  ###################################"""
                experimentManager.logManager.write(
                    "Accuracy on seen data for subject:" + subject + " for Episodic Memory:")
                experimentManager.logManager.write(all_seen_acc_GDM_E)
                # Appending to results for all subjects for the given condition and given order
                all_seen_acc_GDM_E_all_subjects.append(all_seen_acc_GDM_E)
                experimentManager.logManager.write(
                    "Accuracy on seen data for subject:" + subject + " for Semantic Memory:")
                experimentManager.logManager.write(all_seen_acc_GDM_S)
                # Appending to results for all subjects for the given condition and given order
                all_seen_acc_GDM_S_all_subjects.append(all_seen_acc_GDM_S)
                """##################################    F1 Scores on all seen Data #################################"""

                experimentManager.logManager.write(
                    "F1 Score on seen data for subject:" + subject + " for Episodic Memory:")
                experimentManager.logManager.write(all_seen_f1_GDM_E)
                # Appending to results for all subjects for the given condition and given order
                all_seen_f1_GDM_E_all_subjects.append(all_seen_f1_GDM_E)
                experimentManager.logManager.write(
                    "F1 Score on seen data for subject:" + subject + " for Semantic Memory:")
                experimentManager.logManager.write(all_seen_f1_GDM_S)
                # Appending to results for all subjects for the given condition and given order
                all_seen_f1_GDM_S_all_subjects.append(all_seen_f1_GDM_S)

                """##################################    Overall Scores on all Data  ################################"""
                experimentManager.logManager.write("Overall accuracy for subject:" + subject + " for Episodic Memory:")
                experimentManager.logManager.write(overall_acc_GDM_E)
                # Appending to results for all subjects for the given condition and given order
                overall_acc_GDM_E_all_subjects.append(overall_acc_GDM_E)
                experimentManager.logManager.write("Overall accuracy for subject:" + subject + " for Semantic Memory:")
                # Appending to results for all subjects for the given condition and given order
                experimentManager.logManager.write(overall_acc_GDM_S)
                overall_acc_GDM_S_all_subjects.append(overall_acc_GDM_S)

                experimentManager.logManager.write("Overall F1 score for subject:" + subject + " for Episodic Memory:")
                experimentManager.logManager.write(overall_f1_GDM_E)
                # Appending to results for all subjects for the given condition and given order
                overall_f1_GDM_E_all_subjects.append(overall_f1_GDM_E)
                experimentManager.logManager.write("Overall F1 Score for subject:" + subject + " for Semantic Memory:")
                # Appending to results for all subjects for the given condition and given order
                experimentManager.logManager.write(overall_f1_GDM_S)
                overall_f1_GDM_S_all_subjects.append(overall_f1_GDM_S)

                """##################################################################################################"""
                """#############################    Plotting scores for each Subject  ###############################"""
                """##################################################################################################"""

                experimentManager.plotManager.plotMultiAcc(
                    {"Subject: " + subject + "_novel_acc_GDM_E": novel_acc_GDM_E,
                     "Subject: " + subject + "_all_seen_acc_GDM_E": all_seen_acc_GDM_E,
                     "Subject: " + subject + "_overall_acc_GDM_E": overall_acc_GDM_E,
                     "Subject: " + subject + "_novel_acc_GDM_S": novel_acc_GDM_S,
                     "Subject: " + subject + "_all_seen_acc_GDM_S": all_seen_acc_GDM_S,
                     "Subject: " + subject + "_overall_acc_GDM_S": overall_acc_GDM_S},
                    title="Condition: " + str(
                        conditions[cond]) + " Order: " + str(order_label_dictionary),
                    location=save_path + "/" + conditions[cond])

                experimentManager.plotManager.plotMultiF1(
                    {"Subject: " + subject + "_novel_f1_GDM_E": novel_f1_GDM_E,
                     "Subject: " + subject + "_all_seen_f1_GDM_E": all_seen_f1_GDM_E,
                     "Subject: " + subject + "_overall_f1_GDM_E": overall_f1_GDM_E,
                     "Subject: " + subject + "_novel_f1_GDM_S": novel_f1_GDM_S,
                     "Subject: " + subject + "_all_seen_f1_GDM_S": all_seen_f1_GDM_S,
                     "Subject: " + subject + "_overall_f1_GDM_S": overall_f1_GDM_S},
                    title="Condition: " + str(
                        conditions[cond]) + " Order: " + str(order_label_dictionary),
                    location=save_path + "/" + conditions[cond])
                """##################################################################################################"""
                """#############################    Saving Trained models for each subject  #########################"""
                """##################################################################################################"""
                save_models_path = experimentManager.modelDirectory + "/" + str(idx) + "/" + conditions[cond] + \
                                   "/Subject_" + subject
                if not os.path.exists(save_models_path):
                    os.makedirs(save_models_path)
                if not mlp_classifier:
                    gtls.export_network(save_models_path + "/GDM_E", trained_GWRs[0])
                    gtls.export_network(save_models_path + "/GDM_S", trained_GWRs[1])

            """##################################################################################################"""
            """##################    Compiling scores for a given condition for all subjects  ###################"""
            """##################################################################################################"""
            experimentManager.logManager.write("Order :  " + str(order_label_dictionary))
            experimentManager.logManager.write("Condition :  " + conditions[cond])

            # Novel Data Accuracies at each step
            experimentManager.logManager.write("Accuracy on Novel Data for Episodic Memory:")
            novel_acc_GDM_E_all_subjects_per_condition.append(np.array(
                novel_acc_GDM_E_all_subjects).reshape((args.subjects, args.classes)))
            experimentManager.logManager.write(np.array2string(np.array(
                novel_acc_GDM_E_all_subjects).reshape((args.subjects, args.classes)), separator=', '))
            experimentManager.logManager.write("Accuracy on Novel Data for Semantic Memory:")
            novel_acc_GDM_S_all_subjects_per_condition.append(np.array(
                novel_acc_GDM_S_all_subjects).reshape((args.subjects, args.classes)))
            experimentManager.logManager.write(np.array2string(np.array(
                novel_acc_GDM_S_all_subjects).reshape((args.subjects, args.classes)), separator=', '))

            # Novel Data F1 Scores at each step
            experimentManager.logManager.write("F1 Score on Novel Data for Episodic Memory:")
            novel_f1_GDM_E_all_subjects_per_condition.append(np.array(
                novel_f1_GDM_E_all_subjects).reshape((args.subjects, args.classes)))
            experimentManager.logManager.write(np.array2string(np.array(
                novel_f1_GDM_E_all_subjects).reshape((args.subjects, args.classes)), separator=', '))
            experimentManager.logManager.write("F1 Score on Novel Data for Semantic Memory:")
            novel_f1_GDM_S_all_subjects_per_condition.append(np.array(
                novel_f1_GDM_S_all_subjects).reshape((args.subjects, args.classes)))
            experimentManager.logManager.write(np.array2string(np.array(
                novel_f1_GDM_S_all_subjects).reshape((args.subjects, args.classes)), separator=', '))

            # Accuracy on all seen data at each time step
            experimentManager.logManager.write("Accuracy on seen data for Episodic Memory:")
            all_seen_acc_GDM_E_all_subjects_per_condition.append(np.array(
                all_seen_acc_GDM_E_all_subjects).reshape((args.subjects, args.classes)))
            experimentManager.logManager.write(np.array2string(np.array(
                all_seen_acc_GDM_E_all_subjects).reshape((args.subjects, args.classes)), separator=', '))
            experimentManager.logManager.write("Accuracy on seen data for Semantic Memory:")
            all_seen_acc_GDM_S_all_subjects_per_condition.append(np.array(
                all_seen_acc_GDM_S_all_subjects).reshape((args.subjects, args.classes)))
            experimentManager.logManager.write(np.array2string(np.array(
                all_seen_acc_GDM_S_all_subjects).reshape((args.subjects, args.classes)), separator=', '))

            # F1 Score on all seen data at each time step
            experimentManager.logManager.write("F1 Score on seen data for Episodic Memory:")
            all_seen_f1_GDM_E_all_subjects_per_condition.append(np.array(
                all_seen_f1_GDM_E_all_subjects).reshape((args.subjects, args.classes)))
            experimentManager.logManager.write(np.array2string(np.array(
                all_seen_f1_GDM_E_all_subjects).reshape((args.subjects, args.classes)), separator=', '))
            experimentManager.logManager.write("F1 Score on seen data for Semantic Memory:")
            all_seen_f1_GDM_S_all_subjects_per_condition.append(np.array(
                all_seen_f1_GDM_S_all_subjects).reshape((args.subjects, args.classes)))
            experimentManager.logManager.write(np.array2string(np.array(
                all_seen_f1_GDM_S_all_subjects).reshape((args.subjects, args.classes)), separator=', '))

            # Overall accuracy values at each time step
            experimentManager.logManager.write("Overall accuracy for Episodic Memory:")
            overall_acc_GDM_E_all_subjects_per_condition.append(np.array(
                overall_acc_GDM_E_all_subjects).reshape((args.subjects, args.classes)))
            experimentManager.logManager.write(np.array2string(np.array(
                overall_acc_GDM_E_all_subjects).reshape((args.subjects, args.classes)), separator=', '))
            experimentManager.logManager.write("Overall accuracy for Semantic Memory:")
            overall_acc_GDM_S_all_subjects_per_condition.append(np.array(overall_acc_GDM_S_all_subjects))
            experimentManager.logManager.write(np.array2string(np.array(overall_acc_GDM_S_all_subjects),
                                                               separator=', '))

            # Overall F1 Score values at each time step
            experimentManager.logManager.write("Overall F1 Score for Episodic Memory:")
            overall_f1_GDM_E_all_subjects_per_condition.append(np.array(
                overall_f1_GDM_E_all_subjects).reshape((args.subjects, args.classes)))
            experimentManager.logManager.write(np.array2string(np.array(
                overall_f1_GDM_E_all_subjects).reshape((args.subjects, args.classes)), separator=', '))
            experimentManager.logManager.write("Overall F1 Score for Semantic Memory:")
            overall_f1_GDM_S_all_subjects_per_condition.append(np.array(overall_f1_GDM_S_all_subjects))
            experimentManager.logManager.write(np.array2string(np.array(overall_f1_GDM_S_all_subjects),
                                                               separator=', '))

        """#######################################################################################################"""
        """#######################    Generating Plots for all conditions per order ##############################"""
        """#######################################################################################################"""
        # Plotting Accuracies
        plotting_scores(experimentManager, all_seen_acc_GDM_E_all_subjects_per_condition,
                        all_seen_acc_GDM_S_all_subjects_per_condition, overall_acc_GDM_E_all_subjects_per_condition,
                        overall_acc_GDM_S_all_subjects_per_condition, order=order_label_dictionary,
                        location=save_path)
        # Plotting F1 Scores
        plotting_scores(experimentManager, all_seen_f1_GDM_E_all_subjects_per_condition,
                        all_seen_f1_GDM_S_all_subjects_per_condition, overall_f1_GDM_E_all_subjects_per_condition,
                        overall_f1_GDM_S_all_subjects_per_condition, order=order_label_dictionary,
                        location=save_path, f1=True)

        # Appending Accuracy results for all subjects for all conditions per order
        all_seen_acc_GDM_E_all_per_order.append(np.array(
            all_seen_acc_GDM_E_all_subjects_per_condition).reshape((len(conditions), args.subjects, args.classes)))
        all_seen_acc_GDM_S_all_per_order.append(np.array(
            all_seen_acc_GDM_S_all_subjects_per_condition).reshape((len(conditions), args.subjects, args.classes)))
        overall_acc_GDM_E_all_per_order.append(np.array(
            overall_acc_GDM_E_all_subjects_per_condition).reshape((len(conditions), args.subjects, args.classes)))
        overall_acc_GDM_S_all_per_order.append(np.array(
            overall_acc_GDM_S_all_subjects_per_condition).reshape((len(conditions), args.subjects, args.classes)))

        # Appending F1 Score results for all subjects for all conditions per order
        all_seen_f1_GDM_E_all_per_order.append(np.array(
            all_seen_f1_GDM_E_all_subjects_per_condition).reshape((len(conditions), args.subjects, args.classes)))
        all_seen_f1_GDM_S_all_per_order.append(np.array(
            all_seen_f1_GDM_S_all_subjects_per_condition).reshape((len(conditions), args.subjects, args.classes)))
        overall_f1_GDM_E_all_per_order.append(np.array(
            overall_f1_GDM_E_all_subjects_per_condition).reshape((len(conditions), args.subjects, args.classes)))
        overall_f1_GDM_S_all_per_order.append(np.array(
            overall_f1_GDM_S_all_subjects_per_condition).reshape((len(conditions), args.subjects, args.classes)))

    """#######################################################################################################"""
    """#######################    Collating results for all orderings ########################################"""
    """#######################################################################################################"""

    experimentManager.logManager.write("All Seen GDM-E: ")
    all_seen_acc_GDM_E_all_per_order = np.array(all_seen_acc_GDM_E_all_per_order).reshape((len(class_orders),
                                                                                           len(conditions),
                                                                                           args.subjects,
                                                                                           args.classes))

    all_seen_f1_GDM_E_all_per_order = np.array(all_seen_f1_GDM_E_all_per_order).reshape((len(class_orders),
                                                                                         len(conditions),
                                                                                         args.subjects,
                                                                                         args.classes))
    experimentManager.logManager.write(np.array(all_seen_acc_GDM_E_all_per_order).shape)
    experimentManager.logManager.write(np.array2string(np.array(all_seen_acc_GDM_E_all_per_order), separator=', '))
    experimentManager.logManager.write(np.array2string(np.array(all_seen_f1_GDM_E_all_per_order), separator=', '))

    experimentManager.logManager.write("All Seen GDM-S: ")
    all_seen_acc_GDM_S_all_per_order = np.array(all_seen_acc_GDM_S_all_per_order).reshape((len(class_orders),
                                                                                           len(conditions),
                                                                                           args.subjects,
                                                                                           args.classes))

    all_seen_f1_GDM_S_all_per_order = np.array(all_seen_f1_GDM_S_all_per_order).reshape((len(class_orders),
                                                                                         len(conditions),
                                                                                         args.subjects,
                                                                                         args.classes))
    experimentManager.logManager.write(np.array(all_seen_acc_GDM_S_all_per_order).shape)
    experimentManager.logManager.write(np.array2string(np.array(all_seen_acc_GDM_S_all_per_order), separator=', '))
    experimentManager.logManager.write(np.array2string(np.array(all_seen_f1_GDM_S_all_per_order), separator=', '))

    experimentManager.logManager.write("Overall GDM-E: ")
    overall_acc_GDM_E_all_per_order = np.array(overall_acc_GDM_E_all_per_order).reshape((len(class_orders),
                                                                                         len(conditions),
                                                                                         args.subjects,
                                                                                         args.classes))

    overall_f1_GDM_E_all_per_order = np.array(overall_f1_GDM_E_all_per_order).reshape((len(class_orders),
                                                                                       len(conditions),
                                                                                       args.subjects,
                                                                                       args.classes))
    experimentManager.logManager.write(np.array(overall_acc_GDM_E_all_per_order).shape)
    experimentManager.logManager.write(np.array2string(np.array(overall_acc_GDM_E_all_per_order), separator=', '))
    experimentManager.logManager.write(np.array2string(np.array(overall_f1_GDM_E_all_per_order), separator=', '))

    experimentManager.logManager.write("Overall GDM-S: ")
    overall_acc_GDM_S_all_per_order = np.array(overall_acc_GDM_S_all_per_order).reshape((len(class_orders),
                                                                                         len(conditions),
                                                                                         args.subjects,
                                                                                         args.classes))

    overall_f1_GDM_S_all_per_order = np.array(overall_f1_GDM_S_all_per_order).reshape((len(class_orders),
                                                                                       len(conditions),
                                                                                       args.subjects,
                                                                                       args.classes))
    experimentManager.logManager.write(np.array(overall_acc_GDM_S_all_per_order).shape)
    experimentManager.logManager.write(np.array2string(np.array(overall_acc_GDM_S_all_per_order), separator=', '))
    experimentManager.logManager.write(np.array2string(np.array(overall_f1_GDM_S_all_per_order), separator=', '))

    """#######################################################################################################"""
    """################    Computing stats and plotting distributions across all GDM results #################"""
    """#######################################################################################################"""
    # if not mlp_classifier:
    #    compute_stats(experimentManager, class_orders, subjects, class_dictionary, all_seen_acc_GDM_E_all_per_order,
    #                  all_seen_acc_GDM_S_all_per_order, overall_acc_GDM_E_all_per_order, overall_acc_GDM_S_all_per_order,
    #                  all_seen_f1_GDM_E_all_per_order, all_seen_f1_GDM_S_all_per_order, overall_f1_GDM_E_all_per_order,
    #                  overall_f1_GDM_S_all_per_order, args)

    """#######################################################################################################"""
    """########################################## END ########################################################"""
    """#######################################################################################################"""



def parseArguments():
    parser = argparse.ArgumentParser(description="ContinualAffectiveLearning_.")
    parser.add_argument('-e',   '--epochs',       default=100,        type=int)
    parser.add_argument('-b',   '--batch_size',   default=64,         type=int)
    parser.add_argument('-i',   '--imagesize',    default=96,         type=int, help="Image Size NxN")
    parser.add_argument('-l',   '--learningrate', default=0.002,     type=float, help="LearningRate?")
    parser.add_argument('-x',   '--imageresize',  default=True,       type=bool, help="Resize Image?")
    parser.add_argument('-g',   '--grayscale',    default=False,      type=bool, help="Grayscale Image?")
    parser.add_argument('-z',   '--zshape',       default=50,         type=int, help="Latent Variable Size?")
    parser.add_argument('-d',   '--dataset',      default="RAV",      type=str, help='Dataset?')
    parser.add_argument('-a',   '--augment',      default=False,      type=bool, help='Use Augmentation?')
    parser.add_argument('-s',   '--server',       default=False,      type=bool, help='On Server?')
    parser.add_argument('-c',   '--classes',      default=6,          type=int, help='Number of classes?')
    parser.add_argument('-t',   '--taskIL',       default=False,    type=bool, help='TASK-IL?')
    parser.add_argument('-p',   '--subjects',     default=1,          type=int, help='Number of subjects?')
    parser.add_argument('-v',   '--vgg',          default=False,      type=bool, help='VGG-Face?')
    parser.add_argument('-o',   '--order',        default=0,          type=int, help='Order?')
    return parser.parse_args()

if __name__ == "__main__":

    args = parseArguments()

    if args.dataset.startswith("RAV"):
        dataDirectory = "experiment/RAVDESS_Experiments"
        datasetTrainFolder = "data/Ravdess/train/"
        datasetTestFolder = "data/Ravdess/test/"

    elif args.dataset.startswith("MMI"):
        dataDirectory = "experiment/MMI_Experiments"
        datasetTrainFolder = "data/MMI/train/"
        datasetTestFolder = "data/MMI/test/"

    elif args.dataset.startswith("BAUM"):
        dataDirectory = "experiment/BAUM_Experiments"
        datasetTrainFolder = "data/BAUM/train/"
        datasetTestFolder = "data/BAUM/test/"
    else:
        print("Please define a dataset!")
        exit(0)

    imageSize = (args.imagesize, args.imagesize)
    fps = 30
    preProcessingProperties = []
    preProcessingProperties.append(imageSize)
    preProcessingProperties.append(args.imageresize)
    preProcessingProperties.append(args.grayscale)
    preProcessingProperties.append(args.zshape)
    preProcessingProperties.append(fps//2)
    preProcessingProperties.append(args.batch_size)

    experimentManager = ExperimentManager.ExperimentManager(dataDirectory, "Experiments_Imagination_Framework_Order_" + str(args.order)+"_",
                                                                verbose=True)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    with tf.device('/device:GPU:0'):
        run(experimentManager, args, preProcessingProperties, datasetTrainFolder, labelCSV_train)
