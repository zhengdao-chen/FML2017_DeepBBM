import numpy as np
from sklearn import preprocessing
import sklearn.svm as sksvm
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.tree.tree import DecisionTreeClassifier
from scipy.special import comb
import math
from scipy.special import erf
from matplotlib.legend_handler import HandlerLine2D


def fetch_data_from_raw(filename):
    data = np.loadtxt(filename, delimiter=',')
    data_train_feature = data[:3000, :57]
    data_train_label = data[:3000, 57]
    data_test_feature = data[3000:, :57]
    data_test_label = data[3000:, 57]
    scaler = preprocessing.MinMaxScaler(feature_range = (-1, 1))
    scaler.fit_transform(data_train_feature)
    data_train_feature = scaler.transform(data_train_feature)
    data_test_feature = scaler.transform(data_test_feature)
    return data_train_feature, data_train_label, data_test_feature, data_test_label

def fetch_npy_data(filename):
    data_train_feature = np.load(filename + '_features_train.npy')
    data_train_label = np.load(filename + '_labels_train.npy')
    data_test_feature = np.load(filename + '_features_test.npy')
    data_test_label = np.load(filename + '_labels_test.npy')
    scaler = preprocessing.MinMaxScaler(feature_range = (-1, 1))
    scaler.fit_transform(data_train_feature)
    data_train_feature = scaler.transform(data_train_feature)
    data_test_feature = scaler.transform(data_test_feature)
    return data_train_feature, data_train_label, data_test_feature, data_test_label

def convert_to_pm1(labels):
    return labels * 2 - 1


def AdaBoostClf(features, labels, max_depth, n_steps):
    sample_size = features.shape[0]
    weights = np.ones(sample_size) / sample_size
    clf_list = []
    for t in range(n_steps):
        clf = DecisionTreeClassifier(max_depth=max_depth)
        clf = clf.fit(features, labels, sample_weight=weights)
        y_predict = clf.predict(features)
        incorrect = y_predict != labels
        # Error fraction
        estimator_error = np.mean(np.average(incorrect, weights=weights, axis=0))
        if (estimator_error >= 0.5):
            break;
        if (estimator_error == 0):
            clf_list = [[clf, 1]]
            break
        step_size = 0.5 * np.log((1 - estimator_error) / estimator_error)
        norm_factor = 2 * pow(estimator_error * (1 - estimator_error), 0.5)

        for i in range(sample_size):
            if (labels[i] == y_predict[i]):
                weights[i] *= np.exp(-step_size) / norm_factor
            else:
                weights[i] *= np.exp(step_size) / norm_factor
        clf_list.append([clf, step_size])
    return clf_list

def testEnsemble(clf_list, testing_features, true_labels, boolean = True):
    y_predict = np.zeros(true_labels.shape[0])
    for pair in clf_list:
        clf = pair[0]
        step_size = pair[1]
        y_predict += step_size * convert_to_pm1(clf.predict(testing_features))

    if boolean:
        y_predict = (np.sign(y_predict) + 1) / 2
    else:
        y_predict = np.sign(y_predict)
    incorrect = y_predict != true_labels
    test_error = np.mean(incorrect)
    print(y_predict - true_labels)
    return test_error

def MarginBoostClf(features, labels, max_depth, n_steps, margin):
    sample_size = features.shape[0]
    weights = np.ones(sample_size) / sample_size
    clf_list = []
    for t in range(n_steps):
        clf = DecisionTreeClassifier(max_depth=max_depth)
        clf = clf.fit(features, labels, sample_weight=weights)
        y_predict = clf.predict(features)
        incorrect = y_predict != labels
        # Error fraction
        estimator_error = np.mean(np.average(incorrect, weights=weights, axis=0))
        if (estimator_error >= 0.5):
            break;
        step_size = 0.5 * (np.log((1 - estimator_error) / estimator_error) + np.log(1 - margin) - np.log(1 + margin))
        norm_factor = 2 * pow(estimator_error * (1 - estimator_error), 0.5)

        for i in range(sample_size):
            if (labels[i] == y_predict[i]):
                weights[i] *= np.exp(-step_size) / norm_factor
            else:
                weights[i] *= np.exp(step_size) / norm_factor
        clf_list.append([clf, step_size])
    return clf_list
        

def get_k_from_gamma(gamma, sample_size):
    return int(round(1 / (2 * pow(gamma, 2)) * np.log(sample_size / 2)))+1

def BoostByMaj(features, labels, max_depth, gamma):
    sample_size = features.shape[0]
    weights = np.ones(sample_size) / sample_size
    counts = np.zeros(sample_size)
    k_pre = get_k_from_gamma(gamma, sample_size)
    k = k_pre
    #k = min(600, k_pre)
    print ('k ', k)
    clf_list = []
    for i in range(k):
        estimator_error = 0.6
        countdown = 10
        while ((estimator_error >= 0.5) and (countdown >= 0)):
            clf = DecisionTreeClassifier(max_depth=max_depth)
            clf = clf.fit(features, labels, sample_weight=weights)
            y_predict = clf.predict(features)
            correct_ones = y_predict == labels
            incorrect_ones = y_predict != labels
            estimator_error = np.mean(np.average(incorrect_ones, weights=weights, axis=0))
            unweighted_estimator_error = np.mean(np.average(incorrect_ones, axis=0))
            countdown -= 1
        counts += correct_ones
        coeff_1 = int(np.floor(k/2))-counts
        coeff_2 = int(np.ceil(k/2))-i-1+counts
        weights = comb(k-i-1, coeff_1) * pow(0.5+gamma, coeff_1) * pow(0.5-gamma, coeff_2)

        print ('i', i, 'error', estimator_error, 'unweighted_error', unweighted_estimator_error, 'wnorm', np.linalg.norm(weights, ord=1))
        weights = weights / np.linalg.norm(weights, ord=1)
        clf_list.append([clf, 1])
    return clf_list, weights

def calc_rademacher(depth, sample_size, num_features, normalizer):
    rademacher = np.sqrt(((2 * pow(2, depth) + 1) * (np.log(num_features + 2) / np.log(2)) *
            np.log(sample_size)) / sample_size)
    return rademacher


def DeepBBM(features, labels, gamma, max_depth_range, PARAM_lambda_2):
    num_features = features.shape[1]
    sample_size = features.shape[0]
    weights = np.ones(sample_size) / sample_size
    counts = np.zeros(sample_size)
    k_pre = get_k_from_gamma(gamma, sample_size)
    k = k_pre
    #k = min(600, k_pre)
    normalizer = np.exp(1) * sample_size
    # print ('k ', k)
    clf_list = []
    rademacher_list = []
    for depth_index in range(len(max_depth_range)):
        depth = max_depth_range[depth_index]
        rademacher_list.append(calc_rademacher(depth, sample_size, num_features, normalizer))
    for t in range(k):
        best_loss = 10000
        best_error = 1
        best_depth = -1
        best_clf = DecisionTreeClassifier(max_depth=0)
        for depth_index in range(len(max_depth_range)):
            depth = max_depth_range[depth_index]
            new_clf = DecisionTreeClassifier(max_depth=depth)
            new_clf = new_clf.fit(features, labels, sample_weight=weights)
            new_error = eval_clf(new_clf, features, labels, weights)
            new_edge = new_error - 0.5
            new_sign_edge = np.sign(new_edge)
            new_loss = new_error + PARAM_lambda_2 * rademacher_list[depth_index]
            if (new_loss < best_loss):
                best_clf = new_clf
                best_loss = new_loss
                best_error = new_error
                best_depth = depth
        
        y_predict = best_clf.predict(features)
        correct_ones = y_predict == labels
        counts += correct_ones
        coeff_1 = int(np.floor(k/2))-counts
        coeff_2 = int(np.ceil(k/2))-t-1+counts
        weights = comb(k-t-1, coeff_1) * pow(0.5+gamma, coeff_1) * pow(0.5-gamma, coeff_2)

        clf_list.append([best_clf, 1, best_depth])
        # print ('i', t, 'error', best_error, 'wnorm', np.linalg.norm(weights, ord=1))

        if (np.max(coeff_1) < 0):
            break

        weights = weights / np.linalg.norm(weights, ord=1)
        
    return clf_list, weights

def DeepBBM2(features, labels, max_depth, gamma, max_depth_range):
    num_features = features.shape[1]
    sample_size = features.shape[0]
    weights = np.ones(sample_size) / sample_size
    D_weights = np.ones(sample_size) / sample_size
    counts = np.zeros(sample_size)
    k_pre = get_k_from_gamma(gamma, sample_size)
    k = k_pre
    #k = min(600, k_pre)
    normalizer = np.exp(1) * sample_size
    print ('k ', k)
    clf_list = []
    rademacher_list = []
    for depth in max_depth_range:
        rademacher_list.append(calc_rademacher(depth, sample_size, num_features, normalizer))
    for t in range(k):
        best_loss = 10000
        best_error = 1
        best_depth = -1
        best_clf = DecisionTreeClassifier(max_depth=0)
        for depth in max_depth_range:
            new_clf_list, new_weights = DeepBoost(features, labels, 1, max_depth_range, initial_weights=weights)


            new_clf = DecisionTreeClassifier(max_depth=depth)
            new_clf = new_clf.fit(features, labels, sample_weight=weights)
            new_error = eval_clf(new_clf, features, labels, weights)
            new_edge = new_error - 0.5
            new_sign_edge = np.sign(new_edge)
            new_loss = new_error + PARAM_lambda_2 * rademacher_list[depth-1]
#             print ('new_error', new_error, 'new_grad', new_grad)
            print ('depth', depth, 'new_error', new_error, 'new_grad', new_loss)
            if (new_loss < best_loss):
                best_clf = new_clf
                best_loss = new_loss
                best_error = new_error
                best_depth = depth
        
        y_predict = best_clf.predict(features)
        correct_ones = y_predict == labels
        counts += correct_ones
#         if (best_error >= 0.5):
#             break;
        coeff_1 = int(np.floor(k/2))-counts
        coeff_2 = int(np.ceil(k/2))-t-1+counts
        weights = comb(k-t-1, coeff_1) * pow(0.5+gamma, coeff_1) * pow(0.5-gamma, coeff_2)

        print ('i', t, 'error', best_error, 'wnorm', np.linalg.norm(weights, ord=1))
        weights = weights / np.linalg.norm(weights, ord=1)
        
        clf_list.append([best_clf, 1])
    return clf_list, weights

def eval_clf(clf, features, labels, weights):
    y_predict = clf.predict(features)
    incorrect_ones = y_predict != labels
    estimator_error = np.mean(np.average(incorrect_ones, weights=weights, axis=0))
    return estimator_error

def calc_penalty(depth, sample_size, num_features, normalizer, PARAM_lambda, PARAM_beta):
    rademacher = np.sqrt(((2 * pow(2, depth) + 1) * (np.log(num_features + 2) / np.log(2)) *
            np.log(sample_size)) / sample_size)
    return ((PARAM_lambda * rademacher + PARAM_beta) * sample_size) / (2 * normalizer)
    
def gradient(error, depth, alpha, sign_edge, sample_size, num_features, normalizer, PARAM_lambda, PARAM_beta):
    complexity_penalty = calc_penalty(depth, sample_size, num_features, normalizer, PARAM_lambda, PARAM_beta)
    #print('cp', complexity_penalty, 'normalizer', normalizer)
    edge = error - 0.5
    sign_alpha = np.sign(alpha)
    if (abs(alpha) > kTolerance):
        return (edge + sign_alpha * complexity_penalty)
    elif (abs(edge) <= complexity_penalty):
        return 0
    else:
        return (edge - sign_edge * complexity_penalty)
    
def compute_eta(error, depth, alpha, sample_size, num_features, normalizer, PARAM_lambda, PARAM_beta):
    error = max([error, kTolerance])
    error_term = (1 - error) * np.exp(alpha) - error * np.exp(-alpha)
    complexity_penalty = calc_penalty(depth, sample_size, num_features, normalizer, PARAM_lambda, PARAM_beta)
    ratio = complexity_penalty / error
    if (abs(error_term) <= 2 * complexity_penalty):
        eta = -alpha
    elif (error_term > 2 * complexity_penalty):
        eta = np.log(-ratio + np.sqrt(pow(ratio, 2) + (1 - error) / error))
    else:
        eta = np.log(ratio + np.sqrt(pow(ratio, 2) + (1 - error) / error))
    return eta
    
def DeepBoost(features, labels, n_steps, max_depth_range, PARAM_lambda, PARAM_beta, initial_weights=None):
    num_features = features.shape[1]
    sample_size = features.shape[0]
    if (initial_weights==None):
        weights = np.ones(sample_size) / sample_size
    normalizer = np.exp(1) * sample_size
    clf_list = []  
    for t in range(n_steps):
        best_error = 0
        best_grad = 0
        best_index = -1 #?
        old_tree_is_best = False
        for j in range(len(clf_list)):
            triple = clf_list[j]
            alpha = triple[1]
            if (abs(alpha) >= kTolerance):
                old_clf = triple[0]
                tree_depth = triple[2]
                error = eval_clf(old_clf, features, labels, weights)
                edge = error - 0.5
                sign_edge = np.sign(edge)
                grad = gradient(error, tree_depth, alpha, sign_edge, sample_size, num_features, normalizer, PARAM_lambda, PARAM_beta);
                # print ('depth', tree_depth, 'error', error, 'grad', grad)
                if(abs(grad) > abs(best_grad)):
                    best_grad = grad
                    best_error = error
                    best_index = j
                    old_tree_is_best = True
        best_depth = -1
        for depth in max_depth_range:
            new_clf = DecisionTreeClassifier(max_depth=depth)
            new_clf = new_clf.fit(features, labels, sample_weight=weights)
            new_error = eval_clf(new_clf, features, labels, weights)
            new_edge = new_error - 0.5
            new_sign_edge = np.sign(new_edge)
            new_grad = gradient(new_error, depth, 0, new_sign_edge, sample_size, num_features, normalizer, PARAM_lambda, PARAM_beta)
            if (abs(new_grad) > abs(best_grad)):
                best_new_clf = new_clf
                best_grad = new_grad
                best_error = new_error
                best_depth = depth
                old_tree_is_best = False
        if old_tree_is_best:
            triple = clf_list[best_index]
            alpha = triple[1]
            clf = triple[0]
            depth = triple[2]
            eta = compute_eta(best_error, depth, alpha, sample_size, num_features, normalizer, PARAM_lambda, PARAM_beta)
            clf_list[best_index][1] += eta
        else:
            alpha = 0
            clf = best_new_clf
            depth = best_depth
            #print ('t', t, 'best_error', best_error)
            eta = compute_eta(best_error, depth, alpha, sample_size, num_features, normalizer, PARAM_lambda, PARAM_beta)
            clf_list.append([clf, eta, depth])
        old_normalizer = normalizer
        normalizer = 0
        y_predict = clf.predict(features)
        for i in range(sample_size):
            if (labels[i] == y_predict[i]):
                u = eta
            else:
                u = -eta
            weights[i] *= np.exp(-u)
            normalizer += weights[i]
        weights = weights / normalizer
        normalizer = normalizer * old_normalizer
    return clf_list, weights

def BrownBoost(features, labels, max_depth, total_time):
    sample_size = features.shape[0]
    clf_list = []
    r = np.zeros(sample_size)
    weights = np.array([])
    s = total_time
    #s works as the remaining time with the initial value total_time
    T = total_time
    alpha = 0
    i = 0
    b = np.zeros(sample_size)
    while (s > 0 and i < 200):
        weights = np.exp(-(r + s)**2/total_time)
        weights = weights/(np.sum(weights))
        clf = DecisionTreeClassifier(max_depth=max_depth)
        clf = clf.fit(features, labels, sample_weight=weights)
        y_predict = clf.predict(features)
        incorrect = y_predict != labels
        # Error fraction
        estimator_error = np.mean(np.average(incorrect, weights=weights, axis=0))
        print ('estimator_error is', estimator_error)
        if (estimator_error >= 0.5):
            break;
        for j in range(sample_size):
            if (labels[j] == y_predict[j]):
                b[j] = 1
            else:
                b[j] = -1
        a = r + s
        (t, alpha) = SolveODE(a, b, s, sample_size, T)
        r += alpha * b
        s = s - t
        print (s)
        clf_list.append([clf, alpha])
        i += 1
    return clf_list

def SolveODE(a, b, s, sample_size, T):
    #Initial guess for t and alpha
    t = s/2
    alpha = 0.0
    z = np.array([alpha, t])
    const = -np.ones(sample_size)
    v = np.concatenate((b.reshape(1, sample_size), const.reshape(1, sample_size)), axis=0)
    z_pre = z + 1
    #Here we just make z_pre and z different enough to enter the while loop
    max_iter = 20
    #Set the largest number of iteration to avoid infinite loop
    k = 0
    diff = np.sum(np.abs(z_pre - z))
    #Counter of the number of iterations
    while (diff >= 1e-02 and k < max_iter):
        d = a + z.dot(v)
        w = np.exp(-d**2/T)
        W = np.sum(w)
        U = np.sum(w*d*b)
        B = np.sum(w*b)
        V = np.sum(w*d*b*b)
        E = np.sum(erf(d/np.sqrt(T)) - erf(a/np.sqrt(T)))
        t += (T*B*B + np.sqrt(np.pi*T)*V*E)/(2*(V*W - U*B))
        alpha += (T*W*B + np.sqrt(np.pi*T)*U*E)/(2*(V*W - U*B))
        z_pre = z
        #record the results of z from the previous step
        z = np.array([alpha, t])
        diff = np.sum(np.abs(z_pre - z))
        k += 1
        print (k, diff)
    return (t, alpha)

def add_noise(labels, noise_level):
    n = labels.shape[0]
    noisy_labels = np.zeros(n)
    for i in range(n):
        if (np.random.rand(1) > noise_level):
            noisy_labels[i] = labels[i]
        else:
            noisy_labels[i] = 1 - labels[i]
    return noisy_labels

def train(dataset):
    if (dataset == 'spambase'):
        features, labels, testing_features, true_labels = fetch_data_from_raw('spambase')
    else:
        features, labels, testing_features, true_labels = fetch_npy_data(dataset)

    PARAM_lambda = 0.001
    PARAM_lambda_2 = 0.01
    PARAM_beta = 0.001
    gamma = 0.06
    tree_depth = 15
    depth_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    ## DeepBoost
    T = 200
    clf_list_db, weights = DeepBoost(features, labels, T, depth_range, PARAM_lambda, PARAM_beta)
    train_error_db = testEnsemble(clf_list_db, features, labels)
    test_error_db = testEnsemble(clf_list_db, testing_features, true_labels)

    print ('db done')
    ## Deep BBM

    gamma_list = [0.15, 0.1, 0.08, 0.06]
    lambda_2_list = [1, 0.1, 0.01, 0.001, 0.0001]
    max_depth_list = [1, 2, 3, 5, 10, 15]

    # parameter search for Deep BBM
    # train_errors_dbbm = np.zeros([len(gamma_list), len(lambda_2_list), len(max_depth_list)])
    # test_errors_dbbm = np.zeros([len(gamma_list), len(lambda_2_list), len(max_depth_list)])
    # for i in range(len(gamma_list)):
    #     for j in range(len(lambda_2_list)):
    #         for k in range(len(max_depth_list)):
    #             gamma = gamma_list[i]
    #             lambda_2 = lambda_2_list[j]
    #             max_depth = max_depth_list[k]
    #             depth_range = []
    #             for l in range(max_depth):
    #                 depth_range.append(l+1)
    #             print (depth_range)
    #             clf_list_dbbm, weights = DeepBBM(features, labels, gamma, depth_range, lambda_2)
    #             train_error_dbbm = testEnsemble(clf_list_dbbm, features, labels)
    #             test_error_dbbm = testEnsemble(clf_list_dbbm, testing_features, true_labels)
    #             print ('ga', gamma, 'l2', lambda_2, 'md', max_depth, 'TrErr', train_error_dbbm, 'TeErr', test_error_dbbm)
    #             train_errors_dbbm[i, j, k] = train_error_dbbm
    #             test_errors_dbbm[i, j, k] = test_error_dbbm
    # np.save('TrErr_dbbm_ps_2', train_errors_dbbm)
    # np.save('TeErr_dbbm_ps_2', test_errors_dbbm)

    clf_list_dbbm, weights = DeepBBM(features, labels, gamma, depth_range, PARAM_lambda_2)
    train_error_dbbm = testEnsemble(clf_list_dbbm, features, labels)
    test_error_dbbm = testEnsemble(clf_list_dbbm, testing_features, true_labels)
    print ('dbbm done')

    ## DecisionTreeClassifier
    dtc = DecisionTreeClassifier(max_depth=tree_depth)
    dtc = dtc.fit(features, labels)
    train_pred = dtc.predict(features)
    train_mse_dtc = ((train_pred - labels) ** 2).mean(axis=0)
    test_pred = dtc.predict(testing_features)
    # print (np.concatenate((np.expand_dims(pred, axis=1), np.expand_dims(true_labels, axis=1)), axis=1))
    test_mse_dtc = ((test_pred - true_labels) ** 2).mean(axis=0)

    ## Boost by Majority
    # gamma = 0.1
    clf_list_bbm, weights = BoostByMaj(features, labels, tree_depth, gamma)
    train_error_bbm = testEnsemble(clf_list_bbm, features, labels)
    test_error_bbm = testEnsemble(clf_list_bbm, testing_features, true_labels)
    #PlotMarginDistribution(clf_list_bbm, testing_features, true_labels)
    print ('bbm done')

    # ## AdaBoost
    # T = 200
    clf_list_adb = AdaBoostClf(features, labels, tree_depth, T)
    train_error_adb = testEnsemble(clf_list_adb, features, labels)
    test_error_adb = testEnsemble(clf_list_adb, testing_features, true_labels)
    print ('adb done')
    #PlotMarginDistribution(clf_list_adb, testing_features, true_labels)

    ## MarginBoost (from our homework)
    # T = 200
    margin = pow(2, -6)
    clf_list_mb = MarginBoostClf(features, labels, tree_depth, T, margin)
    train_error_mb = testEnsemble(clf_list_mb, features, labels)
    test_error_mb = testEnsemble(clf_list_mb, testing_features, true_labels)
    print ('mb done')
    #PlotMarginDistribution(clf_list_mgb, testing_features, true_labels)

    ## BrownBoost
    total_time = 100
    clf_list_brown = BrownBoost(features, labels, tree_depth, total_time)
    train_error_brown = testEnsemble(clf_list_brown, features, labels)
    test_error_brown = testEnsemble(clf_list_brown, testing_features, true_labels)
    print ('bb done')

    print ('DeepBoost: train_error', train_error_db)
    print ('DeepBoost: test_error', test_error_db)
    print ('DeepBBM: train_error', train_error_dbbm)
    print ('DeepBBM: test_error', test_error_dbbm)
    print ('decision tree: train_mse', train_mse_dtc)
    print ('decision tree: test_mse', test_mse_dtc)
    print ('BBM: train_error', train_error_bbm)
    print ('BBM: test_error', test_error_bbm)
    print ('AdaBoost: train_error', train_error_adb)
    print ('AdaBoost: test_error', test_error_adb)
    print ('MarginBoost: train_error', train_error_mb)
    print ('MarginBoost: test_error', test_error_mb)
    print ('BrownBoost: train_error', train_error_brown)
    print ('BrownBoost: test_error', test_error_brown)

def train_2():
    features, labels, testing_features, true_labels = fetch_data_from_raw('spambase')
    clf_list_dbbm, weights = DeepBBM(features, labels, 0.08, [1, 3, 5, 7, 10], 0.01)
    train_error_dbbm = testEnsemble(clf_list_dbbm, features, labels)
    test_error_dbbm = testEnsemble(clf_list_dbbm, testing_features, true_labels)
    print ('TrErr', train_error_dbbm, 'TeErr', test_error_dbbm)
    print (len(clf_list_dbbm))

def train_diabetes():
    Boolean = False
    features, labels, testing_features, true_labels = fetch_npy_data('diabetes')
    clf_list_dbbm, weights = DeepBBM(features, labels, 0.1, [1, 2, 3, 4, 5], 0.01)
    train_error_dbbm = testEnsemble(clf_list_dbbm, features, labels, Boolean)
    test_error_dbbm = testEnsemble(clf_list_dbbm, testing_features, true_labels, Boolean)
    print ('DeepBBM: train_error', train_error_dbbm)
    print ('DeepBBM: test_error', test_error_dbbm)
    print ('DeepBBM: num_of_clf', len(clf_list_dbbm))

    T = 500
    tree_depth = 5
    clf_list_adb = AdaBoostClf(features, labels, tree_depth, T)
    train_error_adb = testEnsemble(clf_list_adb, features, labels, Boolean)
    test_error_adb = testEnsemble(clf_list_adb, testing_features, true_labels, Boolean)
    print ('AdaBoost: train_error', train_error_adb)
    print ('AdaBoost: test_error', test_error_adb)
    print ('AdaBoost: num_of_clf', len(clf_list_adb))

def tune_dbbm(dataset):
    # parameter search for Deep BBM
    if (dataset == 'spambase'):
        features, labels, testing_features, true_labels = fetch_data_from_raw('spambase')
    else:
        features, labels, testing_features, true_labels = fetch_npy_data(dataset)
    gamma_list = [0.06]
    lambda_2_list = [0.1, 0.01, 0.001, 0.0001]
    max_depth_list = [3, 5, 10, 15, 20]
    train_errors_dbbm = np.zeros([len(gamma_list), len(lambda_2_list), len(max_depth_list)])
    test_errors_dbbm = np.zeros([len(gamma_list), len(lambda_2_list), len(max_depth_list)])
    num_clf_dbbm = np.zeros([len(gamma_list), len(lambda_2_list), len(max_depth_list)])
    for i in range(len(gamma_list)):
        for j in range(len(lambda_2_list)):
            for k in range(len(max_depth_list)):
                gamma = gamma_list[i]
                lambda_2 = lambda_2_list[j]
                max_depth = max_depth_list[k]
                depth_range = []
                for l in range(max_depth):
                    depth_range.append(l+1)
                print (depth_range)
                clf_list_dbbm, weights = DeepBBM(features, labels, gamma, depth_range, lambda_2)
                train_error_dbbm = testEnsemble(clf_list_dbbm, features, labels)
                test_error_dbbm = testEnsemble(clf_list_dbbm, testing_features, true_labels)
                print ('ga', gamma, 'l2', lambda_2, 'md', max_depth, 'TrErr', train_error_dbbm, 'TeErr', test_error_dbbm)
                train_errors_dbbm[i, j, k] = train_error_dbbm
                test_errors_dbbm[i, j, k] = test_error_dbbm
                num_clf_dbbm[i, j, k] = len(clf_list_dbbm)
    # np.save('TrErr_dbbm_spam_ps', train_errors_dbbm)
    # np.save('TeErr_dbbm_spam_ps', test_errors_dbbm)
    # np.save('NumClf_dbbm_spam_ps', num_clf_dbbm)

def tune_db(dataset):
    # parameter search for Deep BBM
    if (dataset == 'spambase'):
        features, labels, testing_features, true_labels = fetch_data_from_raw('spambase')
    else:
        features, labels, testing_features, true_labels = fetch_npy_data(dataset)
    lambda_list = [0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
    beta_list = [0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
    max_depth_list = [3, 5, 10, 15, 20]
    T_list = [100, 200, 500]
    train_errors_db = np.zeros([len(lambda_list), len(beta_list), len(max_depth_list), len(T_list)])
    test_errors_db = np.zeros([len(lambda_list), len(beta_list), len(max_depth_list), len(T_list)])
    num_clf_db = np.zeros([len(lambda_list), len(beta_list), len(max_depth_list), len(T_list)])
    for i in range(len(lambda_list)):
        for j in range(len(beta_list)):
            for k in range(len(max_depth_list)):
                for T_index in range(len(T_list)):
                    PARAM_lambda = lambda_list[i]
                    PARAM_beta = beta_list[j]
                    max_depth = max_depth_list[k]
                    T = T_list[T_index]
                    depth_range = []
                    for l in range(max_depth):
                        depth_range.append(l+1)
                    print (depth_range)
                    clf_list_db, weights = DeepBoost(features, labels, T, depth_range, PARAM_lambda, PARAM_beta)
                    train_error_db = testEnsemble(clf_list_db, features, labels)
                    test_error_db = testEnsemble(clf_list_db, testing_features, true_labels)
                    print ('l1', PARAM_lambda, 'be', PARAM_beta, 'md', max_depth, 'TrErr', train_error_db, 'TeErr', test_error_db)
                    train_errors_db[i, j, k, T_index] = train_error_db
                    test_errors_db[i, j, k, T_index] = test_error_db
                    num_clf_db[i, j, k, T_index] = len(clf_list_db)
    np.save('TrErr_db_spam_ps', train_errors_db)
    np.save('TeErr_db_spam_ps', test_errors_db)
    np.save('NumClf_db_spam_ps', num_clf_db)

def plotWithDepths():
    errs_adb = np.load('TeErr_adb_spam_ps_1.npy')
    errs_db = np.load('TeErr_db_spam_ps.npy')
    errs_dbbm = np.load('TeErr_dbbm_spam_ps.npy')

    print (errs_adb)
    print (errs_db[0, 0, :, 1])
    print (errs_dbbm[0, 1, :])

    line1, = plt.plot([3, 5, 10, 15, 20], errs_adb, 'r', label='AdaBoost')#, [3, 5, 10, 15, 20], errs_db[0, 0, :, 1], 'y', [3, 5, 10, 15, 20], errs_dbbm[0, 1, :], 'b')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=5)})
    line2, = plt.plot([3, 5, 10, 15, 20], errs_db[0, 0, :, 1], 'y', label='DeepBoost')
    plt.legend(handler_map={line2: HandlerLine2D(numpoints=5)})
    line2, = plt.plot([3, 5, 10, 15, 20], errs_dbbm[0, 1, :], 'b', label='DeepBBM')
    plt.legend(handler_map={line2: HandlerLine2D(numpoints=5)})
    plt.xlabel('Depth of Decision Tree')
    plt.ylabel('Testing Error')
    plt.savefig('plot_depths.pdf')
    plt.show()

def Exp_compareAlgos(dataset, num_trials):
    TrErr_db_spam_ca = np.zeros(num_trials)
    TeErr_db_spam_ca = np.zeros(num_trials)
    NumClf_db_spam_ca = np.zeros(num_trials)
    AvgDep_db_spam_ca = np.zeros(num_trials)

    TrErr_dbbm_spam_ca = np.zeros(num_trials)
    TeErr_dbbm_spam_ca = np.zeros(num_trials)
    NumClf_dbbm_spam_ca = np.zeros(num_trials)
    AvgDep_dbbm_spam_ca = np.zeros(num_trials)

    TrErr_bbm_spam_ca = np.zeros(num_trials)
    TeErr_bbm_spam_ca = np.zeros(num_trials)
    NumClf_bbm_spam_ca = np.zeros(num_trials)

    TrErr_db_spam_ca = np.zeros(num_trials)
    TeErr_db_spam_ca = np.zeros(num_trials)
    NumClf_db_spam_ca = np.zeros(num_trials)

    TrErr_adb_spam_ca = np.zeros(num_trials)
    TeErr_adb_spam_ca = np.zeros(num_trials)
    NumClf_adb_spam_ca = np.zeros(num_trials)

    TrErr_bb_spam_ca = np.zeros(num_trials)
    TeErr_bb_spam_ca = np.zeros(num_trials)
    NumClf_bb_spam_ca = np.zeros(num_trials)
    for i in range(num_trials):

        if (dataset == 'spambase'):
            features, labels, testing_features, true_labels = fetch_data_from_raw('spambase')
        else:
            features, labels, testing_features, true_labels = fetch_npy_data(dataset)

        PARAM_lambda = 0.001
        PARAM_lambda_2 = 0.01
        PARAM_beta = 0.001
        gamma = 0.08
        # tree_depth = 15
        tree_depth = 10
        # depth_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        depth_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ## DeepBoost
        T = 200
        clf_list_db, weights = DeepBoost(features, labels, T, depth_range, PARAM_lambda, PARAM_beta)
        train_error_db = testEnsemble(clf_list_db, features, labels)
        test_error_db = testEnsemble(clf_list_db, testing_features, true_labels)
        avg_depth_db = 0
        for j in range(len(clf_list_db)):
            avg_depth_db += clf_list_db[j][2]
        avg_depth_db = avg_depth_db / len(clf_list_db)
        print ('db done')
        ## Deep BBM

        clf_list_dbbm, weights = DeepBBM(features, labels, gamma, depth_range, PARAM_lambda_2)
        train_error_dbbm = testEnsemble(clf_list_dbbm, features, labels)
        test_error_dbbm = testEnsemble(clf_list_dbbm, testing_features, true_labels)
        avg_depth_dbbm = 0
        for j in range(len(clf_list_dbbm)):
            avg_depth_dbbm += clf_list_dbbm[j][2]
        avg_depth_dbbm = avg_depth_dbbm / len(clf_list_dbbm)
        print ('dbbm done')

        ## Boost by Majority
        # gamma = 0.1
        clf_list_bbm, weights = BoostByMaj(features, labels, tree_depth, gamma)
        train_error_bbm = testEnsemble(clf_list_bbm, features, labels)
        test_error_bbm = testEnsemble(clf_list_bbm, testing_features, true_labels)
        #PlotMarginDistribution(clf_list_bbm, testing_features, true_labels)
        print ('bbm done')

        # ## AdaBoost
        # T = 200
        clf_list_adb = AdaBoostClf(features, labels, tree_depth, T)
        train_error_adb = testEnsemble(clf_list_adb, features, labels)
        test_error_adb = testEnsemble(clf_list_adb, testing_features, true_labels)
        print ('adb done')
        #PlotMarginDistribution(clf_list_adb, testing_features, true_labels)

        ## MarginBoost (from our homework)
        # T = 200
        # margin = pow(2, -6)
        # clf_list_mb = MarginBoostClf(features, labels, tree_depth, T, margin)
        # train_error_mb = testEnsemble(clf_list_mb, features, labels)
        # test_error_mb = testEnsemble(clf_list_mb, testing_features, true_labels)
        # print ('mb done')
        #PlotMarginDistribution(clf_list_mgb, testing_features, true_labels)

        ## BrownBoost
        total_time = 100
        clf_list_brown = BrownBoost(features, labels, tree_depth, total_time)
        train_error_brown = testEnsemble(clf_list_brown, features, labels)
        test_error_brown = testEnsemble(clf_list_brown, testing_features, true_labels)
        print ('bb done')


        print ('DeepBoost: train_error', train_error_db)
        print ('DeepBoost: test_error', test_error_db)
        TrErr_db_spam_ca[i] = train_error_db
        TeErr_db_spam_ca[i] = test_error_db
        NumClf_db_spam_ca[i] = len(clf_list_db)
        AvgDep_db_spam_ca[i] = avg_depth_db

        print ('DeepBBM: train_error', train_error_dbbm)
        print ('DeepBBM: test_error', test_error_dbbm)
        TrErr_dbbm_spam_ca[i] = train_error_dbbm
        TeErr_dbbm_spam_ca[i] = test_error_dbbm
        NumClf_dbbm_spam_ca[i] = len(clf_list_dbbm)
        AvgDep_dbbm_spam_ca[i] = avg_depth_dbbm

        # print ('decision tree: train_mse', train_mse_dtc)
        # print ('decision tree: test_mse', test_mse_dtc)
        print ('BBM: train_error', train_error_bbm)
        print ('BBM: test_error', test_error_bbm)
        TrErr_bbm_spam_ca[i] = train_error_bbm
        TeErr_bbm_spam_ca[i] = test_error_bbm
        NumClf_bbm_spam_ca[i] = len(clf_list_bbm)

        print ('AdaBoost: train_error', train_error_adb)
        print ('AdaBoost: test_error', test_error_adb)
        TrErr_adb_spam_ca[i] = train_error_adb
        TeErr_adb_spam_ca[i] = test_error_adb
        NumClf_adb_spam_ca[i] = len(clf_list_adb)

        # print ('MarginBoost: train_error', train_error_mb)
        # print ('MarginBoost: test_error', test_error_mb)

        print ('BrownBoost: train_error', train_error_brown)
        print ('BrownBoost: test_error', test_error_brown)
        TrErr_bb_spam_ca[i] = train_error_brown
        TeErr_bb_spam_ca[i] = test_error_brown
        NumClf_bb_spam_ca[i] = len(clf_list_brown)

    
    np.save('TrErr_db_spam_ca', TrErr_db_spam_ca)
    np.save('TeErr_db_spam_ca', TeErr_db_spam_ca)
    np.save('NumClf_db_spam_ca', NumClf_db_spam_ca)
    np.save('AvgDep_db_spam_ca', AvgDep_db_spam_ca)

    np.save('TrErr_dbbm_spam_ca', TrErr_dbbm_spam_ca)
    np.save('TeErr_dbbm_spam_ca', TeErr_dbbm_spam_ca)
    np.save('NumClf_dbbm_spam_ca', NumClf_dbbm_spam_ca)
    np.save('AvgDep_dbbm_spam_ca', AvgDep_dbbm_spam_ca)

    np.save('TrErr_bbm_spam_ca', TrErr_bbm_spam_ca)
    np.save('TeErr_bbm_spam_ca', TeErr_bbm_spam_ca)
    np.save('NumClf_bbm_spam_ca', NumClf_bbm_spam_ca)

    np.save('TrErr_adb_spam_ca', TrErr_adb_spam_ca)
    np.save('TeErr_adb_spam_ca', TeErr_adb_spam_ca)
    np.save('NumClf_adb_spam_ca', NumClf_adb_spam_ca)

    np.save('TrErr_bb_spam_ca', TrErr_bb_spam_ca)
    np.save('TeErr_bb_spam_ca', TeErr_bb_spam_ca)
    np.save('NumClf_bb_spam_ca', NumClf_bb_spam_ca)


kTolerance = 0.0001

train()
# train_diabetes()
# train_2()
# tune_dbbm('spambase')
# tune_db('spambase')
# train('spambase')
# plotWithDepths()
# Exp_compareAlgos('spambase', 10)

