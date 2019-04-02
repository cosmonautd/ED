import os
import cv2
import time
import math
import numpy
import scipy
import sklearn
import numexpr
import operator

import matplotlib.pyplot as plt

def read(ctn_path):
    with open(ctn_path) as ctn_file:
        ctn = [list(map(float, line.strip().split())) for line in ctn_file.readlines()]
        ctn = numpy.array(ctn)
    return ctn

def sigmoid(x):
    # return 1 / (1 + numpy.exp(-x))
    return numexpr.evaluate('1 / (1 + exp(-x))')

def tanh(x):
    # return numpy.tanh(x)
    return numexpr.evaluate('tanh(x)')

def show(contour, title="Contour"):
    plt.figure()
    x = [x_ for y_, x_ in contour]
    y = [y_ for y_, x_ in contour]
    colors = [[0,0,0]]
    area = numpy.pi*3
    plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    angle, [vy, vx, cy, cx] = axis(contour)
    # endy = cy + 75 * math.sin(angle)
    # endx = cx + 75 * math.cos(angle)
    # plt.plot([cx, endx], [cy, endy])
    # plt.legend([str(angle)])
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('scaled')
    ax = plt.gca()
    ax.invert_yaxis()

def axis(contour):
    [vy, vx, y, x] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
    return numpy.arctan2(vy, vx)[0], [vy, vx, y, x]

def rotate(contour, theta=None):
    if theta is None: theta, _ = axis(contour)
    c = numpy.mean(contour, axis=0)
    contour = contour - c
    cos, sin = numpy.cos(theta), numpy.sin(theta)
    R = numpy.array(((cos,-sin), (sin, cos)))
    contour = numpy.dot(R, contour.T).T
    contour = contour + c
    return contour

def lcg_weights(p, q, seed, a, b, c):
    W = list()
    W.append(seed)
    for i in range(q*(p+1)-1):
        r = (a*W[i] + b) % c
        if r not in W:
            W.append(r)
        else:
            S = list(set(list(range(c))) - set(W))
            S.sort()
            W.append(S[int(len(S)/2)])
    W = numpy.array(W)
    W = (W - numpy.mean(W))/numpy.std(W, ddof=1)
    W = W.reshape((q, p+1))
    return W

class ContourDescriptor():
    def __init__(self, mode, neurons, params):
        assert type(mode) == str
        assert type(params) == tuple
        if mode == "neighborhood":
            assert all(isinstance(x, int) for x in params)
            assert all(x % 2 == 0 for x in params)
            self.f = self.neighborhood_optmized
            self.neurons = neurons
            self.params = params
            self.W = list()
            for param in self.params:
                self.W.append(lcg_weights(param, self.neurons, self.neurons, param, self.neurons*(param+1), (self.neurons*(param+1))**2))
        elif mode == "contour_portion":
            assert all(isinstance(x, int) for x in params)
            assert all(x in [5,10,15,20,25] for x in params)
            self.f = self.contour_portion
            self.neurons = neurons
            self.params = params
            self.W = list()
            add_ = {5:0, 10:1, 15:2, 20:3, 25:4}
            for param in self.params:
                add__ = add_[param]
                self.W.append(lcg_weights(2, self.neurons, self.neurons + add__, 2, self.neurons*(2+1) + add__, (self.neurons*(2+1))**2))
        elif mode == "ratio":
            assert all(isinstance(x, int) for x in params)
            assert all(x in [5,10,15,20,25] for x in params)
            self.f = self.ratio
            self.neurons = neurons
            self.params = params
            self.W = list()
            add_ = {5:0, 10:1, 15:2, 20:3, 25:4}
            for param in self.params:
                add__ = add_[param]
                self.W.append(lcg_weights(2, self.neurons, self.neurons + add__, 2, self.neurons*(2+1) + add__, (self.neurons*(2+1))**2))
        elif mode == "derivative":
            assert all(isinstance(x, int) for x in params)
            assert all(x % 2 == 0 for x in params)
            self.f = self.derivative
            self.neurons = neurons
            self.params = params
            self.W = list()
            add_ = {2:0, 4:1, 6:2, 8:3, 10:4}
            for param in self.params:
                add__ = add_[param]
                self.W.append(lcg_weights(2, self.neurons, self.neurons + add__, 2, self.neurons*(2+1) + add__, (self.neurons*(2+1))**2))
        elif mode == "mahalanobis":
            assert all(isinstance(x, int) for x in params)
            assert all(x % 2 == 0 for x in params)
            self.f = self.mahalanobis
            self.neurons = neurons
            self.params = params
            self.W = list()
            for param in self.params:
                param = 6*param
                self.W.append(lcg_weights(param, self.neurons, self.neurons, param, self.neurons*(param+1), (self.neurons*(param+1))**2))
        elif mode == "angle":
            assert all(isinstance(x, int) for x in params)
            assert all(x in [5,10,15,20,25] for x in params)
            self.f = self.angle_optimized
            self.neurons = neurons
            self.params = params
            self.W = list()
            add_ = {5:0, 10:1, 15:2, 20:3, 25:4}
            for param in self.params:
                add__ = add_[param]
                in__ = 3
                self.W.append(lcg_weights(in__, self.neurons, self.neurons + add__, in__, self.neurons*(in__+1) + add__, (self.neurons*(in__+1))**2))
        elif mode == "angle_plus":
            assert all(isinstance(x, int) for x in params)
            assert all(x in [5,10,15,20,25] for x in params)
            self.f = self.angle_plus_optimized
            self.neurons = neurons
            self.params = params
            self.W = list()
            add_ = {5:0, 10:1, 15:2, 20:3, 25:4}
            for param in self.params:
                add__ = add_[param]
                in__ = 6
                self.W.append(lcg_weights(in__, self.neurons, self.neurons + add__, in__, self.neurons*(in__+1) + add__, (self.neurons*(in__+1))**2))
        else:
            raise ValueError("Mode %s is unavailable" % mode)

    def generate_contour(self, image):
        raise NotImplementedError("Contour generation not yet implemented")
    
    def extract_contour_features(self, image=None, contour=None):
        if image is None and contour is None:
            raise ValueError("At least one image or contour must be provided")
        elif image is not None and contour is None:
            contour = self.generate_contour(image)
        return self.f(contour)
    
    def normalization(self, X):
        # X = sklearn.preprocessing.scale(X, axis=0)
        X = (X - numpy.mean(X, axis=0))/numpy.std(X, axis=0, ddof=1)
        return X

    def description(self, X, D, i):
        X = X.T
        bias_X = -1*numpy.ones(X.shape[1]).reshape(1, X.shape[1])
        X = numpy.concatenate((bias_X, X))
        Z = sigmoid(numpy.dot(self.W[i], X))
        bias_Z = -1*numpy.ones(Z.shape[1]).reshape(1, Z.shape[1])
        Z = numpy.concatenate((bias_Z, Z))
        return numpy.dot((numpy.dot(D,Z.T)),(numpy.linalg.pinv(numpy.dot(Z,Z.T))))
    
    def neighborhood(self, contour):
        c = numpy.mean(contour, axis=0)
        len_ = len(contour)
        M = list()
        for i, r in enumerate(self.params):
            X = numpy.zeros((r*len_,2))
            counter = 0
            r = int(r/2)
            for j in range(len_):
                for k in range(j-r, j+r+1, 1):
                    if j != k:
                        X[counter] = contour[k%len_]
                        counter += 1
            d = numpy.array(contour)
            D = numpy.linalg.norm(d-c, axis=1)/len_
            X = numpy.linalg.norm(X-c, axis=1)/len_
            X = X.reshape((len_, 2*r))
            X = self.normalization(X)
            M.append(self.description(X, D, i))
        return numpy.concatenate(tuple([numpy.array(m).flatten() for m in M]), axis=None)
    
    def neighborhood_optmized(self, contour):
        c = numpy.mean(contour, axis=0)
        len_ = len(contour)
        M = list()
        for i, r in enumerate(self.params):
            X = numpy.zeros((r*len_,1))
            d = numpy.array(contour)
            D = numpy.linalg.norm(d-c, axis=1)/len_
            counter = 0
            r = int(r/2)
            for j in range(len_):
                for k in range(j-r, j+r+1, 1):
                    if j != k:
                        X[counter] = D[k%len_]
                        counter += 1
            X = X.reshape((len_, 2*r))
            X = self.normalization(X)
            M.append(self.description(X, D, i))
        return numpy.concatenate(tuple([numpy.array(m).flatten() for m in M]), axis=None)
    
    def contour_portion(self, contour):
        c = numpy.mean(contour, axis=0)
        M = list()
        for i, r in enumerate(self.params):
            X = list()
            D = list()
            shift = int(numpy.round((r/100)*len(contour)))
            for j, p1 in enumerate(contour):
                p2 = contour[(j+shift) % len(contour)]
                X.append([numpy.linalg.norm(p1-c), numpy.linalg.norm(p2-c)])
                D.append(numpy.linalg.norm(p1-p2))
            X = numpy.array(X)/len(contour)
            D = numpy.array(D)/len(contour)
            X = self.normalization(X)
            M.append(self.description(X, D, i))
        return numpy.concatenate(tuple([numpy.array(m).flatten() for m in M]), axis=None)
    
    def mahalanobis(self, contour):
        c = numpy.mean(contour, axis=0)
        M = list()
        for i, r in enumerate(self.params):
            X = list()
            D = list()
            r = 3*r
            for j in range(len(contour)):
                x = list()
                range_ = [n_ % len(contour) for n_ in range(j-r, j+r+1, 1)]
                for k in range_:
                    if k != j:
                        p = contour[k]
                        x.append(numpy.linalg.norm(p-c))
                X.append(x)
                v1 = contour.take(range_[:int((1/3)*(len(range_)-1))], axis=0)
                v2 = contour.take(range_[int((2/3)*(len(range_)-1))+1:], axis=0)
                v3 = numpy.concatenate((v1,v2), axis=0)
                cov = numpy.cov(v3, rowvar=False)
                p = contour[j % len(contour)]
                mahalanobis = scipy.spatial.distance.mahalanobis(p, numpy.mean(v3, axis=0), cov)
                D.append(mahalanobis)
            X = numpy.array(X)
            D = numpy.array(D)
            X = self.normalization(X)
            M.append(self.description(X, D, i))
        return numpy.concatenate(tuple([numpy.array(m).flatten() for m in M]), axis=None)
    
    def ratio(self, contour):
        c = numpy.mean(contour, axis=0)
        M = list()
        for i, r in enumerate(self.params):
            X = list()
            D = list()
            portion = int((r/100)*len(contour))
            for j in range(len(contour)):
                x_ = list()
                x_.append(numpy.linalg.norm(contour[j] - c))
                x_.append(numpy.linalg.norm(contour[(j+portion)%len(contour)] - c))
                X.append(x_)
                D.append(numpy.linalg.norm(contour[j] - contour[(j+portion)%len(contour)])/(portion+1))
            X = numpy.array(X)/len(contour)
            D = numpy.array(D)
            X = self.normalization(X)
            M.append(self.description(X, D, i))
        return numpy.concatenate(tuple([numpy.array(m).flatten() for m in M]), axis=None)
    
    def derivative(self, contour):
        c = numpy.mean(contour, axis=0)
        M = list()
        for i, r in enumerate(self.params):
            X = list()
            r = int(r/2)
            for j in range(len(contour)):
                X.append(contour[(j+r)%len(contour)] - contour[(j-r)%len(contour)])
            X = numpy.array(X)/len(contour)
            d = numpy.array(contour)
            D = numpy.linalg.norm(d-c, axis=1)/len(contour)
            X = self.normalization(X)
            M.append(self.description(X, D, i))
        return numpy.concatenate(tuple([numpy.array(m).flatten() for m in M]), axis=None)
    
    def angle(self, contour):
        c = numpy.mean(contour, axis=0)
        M = list()
        for i, r in enumerate(self.params):
            X = list()
            D = list()
            shift = int(numpy.round((r/100)*len(contour)))
            for j, p1 in enumerate(contour):
                p2 = contour[(j+int(numpy.round(shift/2))) % len(contour)]
                p3 = contour[(j+shift) % len(contour)]
                a_ = numpy.linalg.norm(p2-p3)
                b_ = numpy.linalg.norm(p2-p1)
                c_ = numpy.linalg.norm(p3-p1)
                X.append([numpy.linalg.norm(p1-c), numpy.linalg.norm(p2-c), numpy.linalg.norm(p3-c)])
                D.append(numpy.arccos((a_**2 + b_**2 - c_**2)/(2*a_*b_)))
            X = numpy.array(X)/len(contour)
            D = numpy.array(D)
            D = numpy.nan_to_num(D)
            X = self.normalization(X)
            M.append(self.description(X, D, i))
        return numpy.concatenate(tuple([numpy.array(m).flatten() for m in M]), axis=None)
    
    def angle_optimized(self, contour):
        c = numpy.mean(contour, axis=0)
        len_ = len(contour)
        M = list()
        for i, r in enumerate(self.params):
            p1 = contour
            p2 = numpy.zeros((len_, 2))
            p3 = numpy.zeros((len_, 2))
            shift = int(numpy.round((r/100)*len_))
            shift_2 = int(numpy.round(shift/2))
            for j in range(len_):
                p2[j] = contour[(j+shift_2) % len_]
                p3[j] = contour[(j+shift) % len_]
            P = numpy.zeros((len_, 3, 2))
            P[:,0,:] = p1
            P[:,1,:] = p2
            P[:,2,:] = p3
            P = P.reshape((3*len_,2))
            X = numpy.linalg.norm(P-c, axis=1).reshape(len_, 3)/len_
            a_ = numpy.linalg.norm(p2-p3, axis=1)
            b_ = numpy.linalg.norm(p2-p1, axis=1)
            c_ = numpy.linalg.norm(p3-p1, axis=1)
            D = numpy.arccos((a_**2 + b_**2 - c_**2)/(2*a_*b_))
            D = numpy.nan_to_num(D)
            X = self.normalization(X)
            M.append(self.description(X, D, i))
        return numpy.concatenate(tuple([numpy.array(m).flatten() for m in M]), axis=None)

    def angle_plus(self, contour):
        c = numpy.mean(contour, axis=0)
        M = list()
        for i, r in enumerate(self.params):
            X = list()
            D = list()
            shift = int(numpy.round((r/100)*len(contour)))
            for j, p1 in enumerate(contour):
                p2 = contour[(j+int(numpy.round(shift/2))) % len(contour)]
                p3 = contour[(j+shift) % len(contour)]
                a_ = numpy.linalg.norm(p2-p3)
                b_ = numpy.linalg.norm(p2-p1)
                c_ = numpy.linalg.norm(p3-p1)
                X.append([numpy.linalg.norm(p1-c), numpy.linalg.norm(p2-c), numpy.linalg.norm(p3-c), a_, b_, c_])
                D.append(numpy.arccos((a_**2 + b_**2 - c_**2)/(2*a_*b_)))
            X = numpy.array(X)/len(contour)
            D = numpy.array(D)
            D = numpy.nan_to_num(D)
            X = self.normalization(X)
            M.append(self.description(X, D, i))
        return numpy.concatenate(tuple([numpy.array(m).flatten() for m in M]), axis=None)
    
    def angle_plus_optimized(self, contour):
        c = numpy.mean(contour, axis=0)
        len_ = len(contour)
        M = list()
        for i, r in enumerate(self.params):
            p1 = contour
            p2 = numpy.zeros((len_, 2))
            p3 = numpy.zeros((len_, 2))
            shift = int(numpy.round((r/100)*len_))
            shift_2 = int(numpy.round(shift/2))
            for j in range(len_):
                p2[j] = contour[(j+shift_2) % len_]
                p3[j] = contour[(j+shift) % len_]
            P = numpy.zeros((len_, 3, 2))
            P[:,0,:] = p1
            P[:,1,:] = p2
            P[:,2,:] = p3
            P = P.reshape((3*len_,2))
            X = numpy.linalg.norm(P-c, axis=1).reshape(len_, 3)/len_
            a_ = numpy.linalg.norm(p2-p3, axis=1)
            b_ = numpy.linalg.norm(p2-p1, axis=1)
            c_ = numpy.linalg.norm(p3-p1, axis=1)
            X = numpy.concatenate((X, a_.reshape((len_,1))), axis=1)
            X = numpy.concatenate((X, b_.reshape((len_,1))), axis=1)
            X = numpy.concatenate((X, c_.reshape((len_,1))), axis=1)
            D = numpy.arccos((a_**2 + b_**2 - c_**2)/(2*a_*b_))
            D = numpy.nan_to_num(D)
            X = self.normalization(X)
            M.append(self.description(X, D, i))
        return numpy.concatenate(tuple([numpy.array(m).flatten() for m in M]), axis=None)

class StackedContourDescriptor():
    def __init__(self, descriptors):
        assert type(descriptors) == list
        assert all(isinstance(x, ContourDescriptor) for x in descriptors)
        self.descriptors = descriptors
    
    def generate_contour(self, image):
        # image = 255 - image # fix for images whose object is black with white background (need to normalize)
        contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(contours) < 1: raise ValueError("Can't find contours")
        main_contour = max(contours, key=lambda x:len(x))
        main_contour = numpy.reshape(main_contour, (len(main_contour), 2))
        main_contour = main_contour.astype(float)
        return main_contour
    
    def extract_contour_features(self, image=None, contour=None):
        if image is None and contour is None:
            raise ValueError("At least one image or contour must be provided")
        elif image is not None and contour is None:
            contour = self.generate_contour(image)
        return numpy.concatenate(tuple([d.extract_contour_features(contour=contour) for d in self.descriptors]), axis=None)