import csv, math, os
import numpy as np
from collections import OrderedDict

import matplotlib.pyplot as plt
from matplotlib.image import imread
from scipy.linalg import svd
from scipy.fftpack import dct, fft2
from scipy import ndimage

from PyML import VectorDataSet, SVM, ker
from PyML.classifiers import multi

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

class JAFFEImage:
    def __init__(self, imgPath):
        self.imgPath = imgPath

        # Parsing the title to get the information related to the image
        info = self.parseTitle()
        self.subjName = info["name"]
        self.emotion = info["emotion"]
        self.imgId = info["imgId"]
        self.format = info["format"]

        # Processing Image
        self.imageMatrix = self.processImage()

        # Caching 
        self._zigzagIndices = None

    def parseTitle(self):
        results = {}
        title = os.path.basename(os.path.normpath(self.imgPath))
        info = title.split(".")
        results["name"] = info[0] # ex. KA
        results["emotion"] = info[1][0:2] # ex. AN (for anger)
        results["imgId"] = info[2]  # ex. '39'
        results["format"] = info[3] # ex. tiff
        return results

    def processImage(self):
        imageMatrix = imread(self.imgPath)
        if len(imageMatrix.shape) == 3:
            imageMatrix = rgb2gray(imageMatrix) #always return 2D greyscale image

        # Highpass filter 1
        # lowpass = ndimage.gaussian_filter(imageMatrix, 3)
        # gauss_highpass = imageMatrix - lowpass

        # Highpass filter 2
        # kernel = np.array([[-1, -1, -1],
        #            [-1,  8, -1],
        #            [-1, -1, -1]])
        # highpass_3x3 = ndimage.convolve(imageMatrix, kernel)

        return imageMatrix

    def _zigzagIndex(self, n):
        """
            Adopted from http://rosettacode.org/wiki/Zig-zag_matrix#Python
        """
        indexorder = sorted(((x,y) for x in range(n) for y in range(n)),
                        key = lambda (x,y): (x+y, -y if (x+y) % 2 else y) )
        return {index: n for index,n in enumerate(indexorder)}

    def _zigzagMatrix(self, matrix, n):
        """
            Returning the items of the matrix using ZigZag scanning
            matrix has to be 1 or 2 dimensions
            n is the number of items we need to return
        """
        dims = matrix.shape
        if len(dims) == 1:
            if dims[0] >= n:
                return matrix[:n] # 1-D case unchanged
        elif len(dims) == 2 and dims[0]*dims[1] >= n:
            zarray = []
            dimx = dims[0] 
            if self._zigzagIndices == None or len(self._zigzagIndices) != n:
                self._zigzagIndices = self._zigzagIndex(dimx)
            for i in range(0, n):
                index = self._zigzagIndices[i]
                zarray.append(matrix.item(index))
            return zarray

        return None

    def computeFeatures(self, n=12, plot=False, method="fft"):
        """
            n: The max number of features we want to keep. 
               n=128 seems to be good for both DCT and FFT, but is it overfitting? Try split test.
            plot: Whether or not to show the DCT coeff plot
            method: fft, dct
            Return
            {
                feature_name: feature_value
            }
        """
        features = OrderedDict()

        coeffs = None
        if method == "dct":
            coeffsRaw = dct(self.imageMatrix, norm='ortho') 
            # We need to zigzag the coefficients here, see R plot
            coeffs = self._zigzagMatrix(coeffsRaw, n)
            #coeffs = coeffsRaw.flatten()[:n] # flatten to create 1-D feature vector
        else:
            coeffsRaw = np.real(fft2(self.imageMatrix)) # we wants the magnitude
            # coeffs = np.cumsum(np.absolute(coeffsRaw), 0)[-1,]
            #TODO: We need to zigzag the coefficients here, see R plot
            coeffs = self._zigzagMatrix(coeffsRaw, n)
            #coeffs = coeffsRaw.flatten()[:n] # flatten to create 1-D feature vector
        feaCnt = 1
        if n == -1:
            n = len(coeffs)
        for a in coeffs:
            key = "v"+str(feaCnt)
            features[key] = round(a, 8)
            feaCnt+=1
        if(plot):
            print(coeffs)
            xx = range(len(coeffs))
            plt.plot(xx, coeffs)
            plt.show()
        return features

class MLUtilities:

    @staticmethod
    def readImages(basePath):
        files = os.listdir(basePath)
        for f in files:
            if f.endswith(".png"):
                imgPath = os.path.join(basePath, f)
                yield(JAFFEImage(imgPath))

    @staticmethod
    def pyMLSVM(basePath, isMulti = False):
        """
            Run SVM on Image Training Data 
            basePath is the directory containing the training file
            isMulti is True when we are running multiclass classication
        """

        # Creating training dataset as VectorDataSet object
        cnt = 1
        trainLabels = []
        trainRawData = []
        for img in MLUtilities.readImages(basePath):
            if cnt % 1000 == 0:
                print("Creating row "+ str(cnt))
            row = None
            if isMulti:
                row = MLUtilities.createPyMLSVMRow(img)
            else:
                row = MLUtilities.createPyMLBinarySVMRow(img)
            trainLabels.append(row[0])
            trainRawData.append(row[1])
            cnt+=1
        trainData = VectorDataSet(trainRawData, L=trainLabels)

        # Changing Kernel 
        # k = ker.Polynomial(degree = 2)
        # trainData.attachKernel(k)


        r = None
        # Training SVM
        if isMulti:
            m = multi.OneAgainstRest (SVM())
            r = m.cv(trainData, numFolds=5)
        else:
            s = SVM()
            r = s.cv(trainData, numFolds=5)

        print(r.getConfusionMatrix())
        print(r.getROC())

    @staticmethod
    def createPyMLSVMRow(fImage):
        """
            Each facial image object is a row in PyML SVM
        """        
        row = []
        emoStr = fImage.emotion
        features = fImage.computeFeatures(n=128, method="fft")
        for (k, v) in features.iteritems():
            row.append(v)
        return (emoStr, row)

    @staticmethod
    def createPyMLBinarySVMRow(fImage, emotion="HA"):
        """
            Each facial image object is a row in PyML SVM
        """        
        row = []
        emoStr = "1"
        if fImage.emotion != emotion:
            emoStr = "0"
        features = fImage.computeFeatures(n=48, method="dct")
        for (k, v) in features.iteritems():
            row.append(v)
        return (emoStr, row)
