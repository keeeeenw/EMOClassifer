
class KaggleFacialImage:
    def __init__(self, csvRow):
        """
            emotion 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
        """
        self.emotion = csvRow[0]
        self.pixels = self.parsePixelStr(csvRow[1])
        self.usage = csvRow[2]

    def parsePixelStr(self, pixelsStr):
        """
            pixelsStr: '70 80 82 ... 82' of length 2304 (48*48)
            output: array of pixels
        """
        return [float(x) for x in pixelsStr.split()]

    def getMatrixRep(self):
        """
            Return the matrix represenation of the image
            based on self.pixels
            Assume self.pixels is a square image
        """
        dim = int(math.sqrt(len(self.pixels)))
        imageArr = []
        for i in range(dim):
            row = self.pixels[i*48:(i*48+48)]
            imageArr.append(row)
        return np.matrix(imageArr)

    def displayImage(self):
        imageMatrix = self.getMatrixRep()
        plt.imshow(imageMatrix)
        plt.show()

    def saveImage(self, outPath):
        imageMatrix = self.getMatrixRep()
        plt.imsave(outPath, imageMatrix, cmap="gray")

    def computeFeaturesDCT(self, n, plot=False):
        """
            n: The max number of features we want to keep
            plot: Whether or not to show the DCT coeff plot
            Return
            {
                feature_name: feature_value
            }
        """
        features = OrderedDict()
        coeffs = dct(self.pixels)
        feaCnt = 1
        if n == -1:
            n = len(coeffs)
        for a in coeffs[:n]:
            key = "v"+str(feaCnt)
            features[key] = round(a, 5)
            feaCnt+=1
        if(plot):
            print(coeffs)
            xx = range(len(coeffs))
            plt.plot(xx, coeffs)
            plt.show()
        return features



"""
    The following functions are designed for running KaggleFacialImage
"""

def genImages(filename, n=100):
    """
        This class generates the image based on the CSV values
    """
    cnt = 0
    for img in readCSVData(filename):
        name = str(cnt)+".png"
        img.saveImage(name)
        cnt+=1
        if cnt > n:
            break

def readCSVData(filename):
    with open(filename, "rb") as csvfile:
        datareader = csv.reader(csvfile)
        cnt = 0
        for row in datareader:
            cnt+=1
            if cnt != 1: #header
                yield KaggleFacialImage(row)

def pyMLSVM(srcfile, n = -1, emotion = "3"):
    """
        Run SVM on Image Training Data 
        emotion = 3 is happy by default
        n is the number of training data, -1 means the full training set
    """

    # Creating training dataset as VectorDataSet object
    cnt = 1
    trainLabels = []
    trainRawData = []
    testLabels = []
    testRawData = []
    for img in readCSVData(srcfile):
        if n != -1 and cnt > n:
            break
        if cnt % 1000 == 0:
            print("Creating row "+ str(cnt))
        row = createPyMLSVMRow(img, emotion)
        if img.usage == "Training":
            trainLabels.append(row[0])
            trainRawData.append(row[1])
        else:
            testLabels.append(row[0])
            testRawData.append(row[1])
        cnt+=1
    trainData = VectorDataSet(trainRawData, L=trainLabels)
    testData = VectorDataSet(testRawData, L=testLabels)

    # Training SVM
    s = SVM()
    s.train(trainData)
    r = s.test(testData)
    print(r.getConfusionMatrix())
    print(r.getROC())

def createPyMLSVMRow(fImage, emotion):
    """
        Each facial image object is a row in PyML SVM
    """        
    row = []
    emoStr = 0
    if emotion == fImage.emotion:
        emoStr = 1
    features = fImage.computeFeaturesDCT(12)
    for (k, v) in features.iteritems():
        row.append(v)
    return (emoStr, row)

def createVWRow(fImage, emotion):
    """
        Each facial image object is a row in the vw file
    """        
    row = ""
    emoStr = "-1"
    if emotion == fImage.emotion:
        emoStr = "1"
    features = fImage.computeFeaturesDCT(-1)
    featuresStr = " ".join(str(k)+":"+str(v) for (k, v) in features.iteritems())
    row = emoStr +" | "+featuresStr
    return row

def createVWFile(srcfile, emotion = "3", split=False):
    """
        Create vm file as input to vowpal wabbit
        emotion = 3 is happy by default
        split = False do not split the file based on "Training" or "PrivateTest" label
    """

    # write features to vw file 
    if split:
        ftrain = open("isHappyTraining.dat","w")
        ftest = open("isHappyTesting.dat","w")
    else:
        f = open("isHappyFull.dat","w")
    cnt = 1
    for img in readCSVData(srcfile):
        if cnt % 100 == 0:
            print("Creating row "+ str(cnt))
        vwRow = createVWRow(img, emotion) 
        if split: #Splitting the Training and Testing File
            if img.usage == "Training":
                ftrain.write(vwRow+"\n")
            else:
                ftest.write(vwRow+"\n")
        else:
            f.write(vwRow+"\n")
        cnt+=1

    ftrain.close()
    ftest.close()

def createArffRow(fImage, emotion):
    """
        Each facial image object is a row in the arff file
    """        
    row = ""
    emoStr = "em0"
    if emotion == fImage.emotion:
        emoStr = "em1"
    features = fImage.computeFeaturesDCT(48)
    featuresStr = ", ".join(str(x) for x in features.values())
    row = emoStr +", "+featuresStr
    return row

def createArffFile(srcfile):
    """
        Create arff file for weka based 
        the image source file
    """

    features = OrderedDict({"isNeutral":"{1, 0}"})
    
    # Get feature names from first image
    for fImage in readCSVData(srcfile):
        fea_names = fImage.computeFeaturesDCT(48)
        for fn in fea_names:
            features[fn] = "numeric"
        break

    with open("isNeutral.arff","w") as f:
        # creating headers
        f.write("@relation isNeutral \n")

        f.write("\n")

        for fea, ftype in features.iteritems():
            f.write("@attribute %s %s \n"%(fea, ftype))

        f.write("\n")

        # creating data
        f.write("@data\n")
        
        # write features to arff file 
        cnt = 1
        for img in readCSVData(srcfile):
            if cnt % 100 == 0:
                print("Creating row "+ str(cnt))
            arffRow = createArffRow(img, "6") # 3 is happy
            f.write(arffRow+"\n")
            cnt+=1