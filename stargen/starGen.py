import glob
import math
from dataclasses import dataclass
import numpy as np
import pandas as pd
from astropy.io import fits
import random
import os
import time
import cv2
from scipy.stats import poisson
from astropy.modeling.models import Sersic2D

import configuration
from tqdm import tqdm

import matplotlib.pyplot as plt


@dataclass
class BoundingBox:
    minX: int
    maxX: int
    minY: int
    maxY: int

    def merge(self, boundingBox, sizeX, sizeY):
        if boundingBox is None:
            return self

        minX = self.minX + boundingBox.minX
        maxX = sizeX - ((sizeX - self.maxX) + (sizeX - boundingBox.maxX))

        minY = self.minY + boundingBox.minY
        maxY = sizeY - ((sizeY - self.maxY) + (sizeY - boundingBox.maxY))

        return BoundingBox(minX, maxX, minY, maxY)

    def reverse(self, sizeX, sizeY, toleranceX=0, toleranceY=0):
        cornerTolerance = 0
        top = BoundingBox(cornerTolerance, sizeX - cornerTolerance, cornerTolerance, self.minY - toleranceY)
        bottom = BoundingBox(cornerTolerance, sizeX - cornerTolerance, self.maxY + toleranceY, sizeY - cornerTolerance)
        left = BoundingBox(cornerTolerance, self.minX - toleranceX, cornerTolerance, sizeY - cornerTolerance)
        right = BoundingBox(self.maxX + toleranceX, sizeX - cornerTolerance, cornerTolerance, sizeY - cornerTolerance)

        return [top, bottom, left, right]

    def isValid(self):
        return self.minY <= self.maxY and self.minX <= self.maxX


@dataclass
class Star:
    x: float
    y: float
    brightness: int
    fwhm: float
    length: int
    alpha: int

    def toTSV(self):
        return [self.x, self.y, self.brightness, 0]


@dataclass
class Galaxy:
    x: float
    y: float
    brightness: int
    alpha: int
    sigmaX: float
    sigmaY: float
    brightnessFactor: float = 0
    sigmaFactor: float = 0

    def toTSV(self):
        return [self.x, self.y, self.brightness, 0]


@dataclass
class Object:
    x: float
    y: float
    brightness: int
    fwhm: float
    length: int
    alpha: int
    positions: list
    isCluster: bool

    def toTSV(self, pos):
        return [self.positions[pos][0], self.positions[pos][1], self.brightness, 1]


@dataclass
class ObjectWithFile:
    obj: Object
    filename: str


class ObjectDataFactory:

    def __init__(self, objType):
        self.objectType = objType

    def fromData(self, x, y, intensity):
        return ObjectData(
            pixelCount=1,
            xList=[x],
            yList=[y],
            totalIntensity=intensity,
            maxIntensity=intensity,
            minIntensity=intensity,
            objectType=self.objectType
        )

    def fromImage(self, image, centerX=None, centerY=None):
        yList, xList = np.nonzero(image)
        pixelCount = len(xList)

        objImage = image[yList, xList]
        totalIntensity = np.sum(objImage)
        maxIntensity = np.amax(objImage)
        minIntensity = np.amin(objImage)

        return ObjectData(
            pixelCount=pixelCount,
            xList=xList,
            yList=yList,
            totalIntensity=totalIntensity,
            maxIntensity=maxIntensity,
            minIntensity=minIntensity,
            objectType=self.objectType,
            centerX=centerX,
            centerY=centerY
        )


@dataclass
class ObjectData:
    pixelCount: int
    xList: list
    yList: list
    totalIntensity: float
    maxIntensity: float
    minIntensity: float
    objectType: str

    boundingBox: BoundingBox = None
    centerX: float = None
    centerY: float = None

    def getBoundingBox(self):
        if self.boundingBox is None:
            minX = np.amin(self.xList)
            maxX = np.amax(self.xList)
            minY = np.amin(self.yList)
            maxY = np.amax(self.yList)

            self.boundingBox = BoundingBox(minX, maxX, minY, maxY)

        return self.boundingBox

    def toTSV(self):
        cenX = '--' if self.centerX is None else self.centerX
        cenY = '--' if self.centerY is None else self.centerY

        return [self.objectType, cenX, cenY, self.pixelCount, self.totalIntensity, self.maxIntensity]


'''
    util class with helping functions
'''


class Utils:
    config: configuration.Configuration

    def __init__(self, config):
        self.config = config

    def clip(self, image):
        return np.clip(image, 0, 65535)

    def calcPosition(self, x, y, linePointA, linePointB):
        x1 = linePointA[0]
        y1 = linePointA[1]
        x2 = linePointB[0]
        y2 = linePointB[1]
        return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)

    def checkRight(self, x, y, linePointA, linePointB):
        return self.calcPosition(x, y, linePointA, linePointB) >= 0

    def checkLeft(self, x, y, linePointA, linePointB):
        return self.calcPosition(x, y, linePointA, linePointB) <= 0

    def clamp(self, point, max0, min0=0):
        if point < min0:
            return min0
        elif point > max0:
            return max0
        return point

    def getVectors(self, halfLengthA, halfLengthB, alpha):
        vectorA = np.array([halfLengthA * np.cos(alpha), halfLengthA * np.sin(alpha)])
        vectorB = np.array([halfLengthB * np.cos(alpha + np.pi / 2), halfLengthB * np.sin(alpha + np.pi / 2)])

        return vectorA, vectorB

    def getCorners(self, center, halfLengthA, halfLengthB, alpha):

        # basic vectors of the box
        vectorCenter = center
        vectorA, vectorB = self.getVectors(halfLengthA, halfLengthB, alpha)

        # four corners of the box
        TL = vectorCenter - vectorA + vectorB
        TR = vectorCenter + vectorA + vectorB
        BL = vectorCenter - vectorA - vectorB
        BR = vectorCenter + vectorA - vectorB

        # define box pixel borders (four lines perpendicular to axes and passing through box four corners)
        boxTop = np.floor(max(TL[1], TR[1], BL[1], BR[1]))
        boxBottom = np.ceil(min(TL[1], TR[1], BL[1], BR[1]))
        boxLeft = np.ceil(min(TL[0], TR[0], BL[0], BR[0]))
        boxRight = np.floor(max(TL[0], TR[0], BL[0], BR[0]))

        boxTop = int(self.clamp(boxTop, self.config.SizeY - 1))
        boxBottom = int(self.clamp(boxBottom, self.config.SizeY - 1))
        boxLeft = int(self.clamp(boxLeft, self.config.SizeX - 1))
        boxRight = int(self.clamp(boxRight, self.config.SizeX - 1))

        return {'top': boxTop, 'bottom': boxBottom, 'left': boxLeft, 'right': boxRight}

    def projectPointOntoLine(self, point, linePointA, linePointB):
        # project point P onto line given by points A and B
        AP = point - linePointA
        AB = linePointB - linePointA
        return linePointA + (np.dot(AP, AB) / np.dot(AB, AB)) * AB

    def axis(self, r, c, axis):
        if axis.ndim > 1:
            return axis[r, c]
        else:
            return axis[c]

    def randomPosition(self, round=False, boundingBox=None):
        if boundingBox is None:
            x = random.uniform(0, self.config.SizeX - 1)
            y = random.uniform(0, self.config.SizeY - 1)
        else:
            # need to check if boundingBox is valid because uniform doesnt care if min < max
            if not boundingBox.isValid():
                x = self.config.SizeX / 2 + 0.1
                y = self.config.SizeY / 2 + 0.1

            else:
                x = random.uniform(boundingBox.minX, boundingBox.maxX)
                y = random.uniform(boundingBox.minY, boundingBox.maxY)

        if round:
            x, y = np.ceil(x).astype('int'), np.ceil(y).astype('int')

        return np.array([x, y])

    def computeGalaxyBoundingBox(self, sigmaX, sigmaY):
        limX = np.ceil(5 * sigmaX)
        limY = np.ceil(5 * sigmaY)

        return self.createBoundingBoxBySize(limX, limY)

    def computeStreakBoundingBox(self, fwhm, length, alpha):
        halfLengthA, halfLengthB = self.getHalfLengths(self.sigma(fwhm), length)
        vectorA, vectorB = self.getVectors(halfLengthA, halfLengthB, np.deg2rad(alpha))
        boundingBox = self.createBoundingBoxForStreak(vectorA, vectorB)

        return boundingBox

    def computeReversedStreakBoundingBox(self, fwhm, length, alpha):
        boundingBox = self.computeStreakBoundingBox(fwhm, length, alpha)

        toleranceX = boundingBox.minX * (4 / 5 if length < 4 else 3 / 4)
        toleranceY = boundingBox.minY * (4 / 5 if length < 4 else 3 / 4)
        reversedBoundingBoxes = boundingBox.reverse(self.config.SizeX, self.config.SizeY, toleranceX, toleranceY)
        vertical = reversedBoundingBoxes[2:]
        horizontal = reversedBoundingBoxes[:2]

        boundingBoxesList = []
        if alpha < 15 or alpha >= 165:
            boundingBoxesList = vertical
        elif 15 <= alpha < 70 or 110 <= alpha < 165:
            boundingBoxesList = reversedBoundingBoxes
        elif 70 <= alpha < 110:
            boundingBoxesList = horizontal

        reversedBB = np.random.choice(boundingBoxesList)

        return reversedBB

    def movePositionIfInCorner(self, pos, cornerSize=1):
        if pos[0] < cornerSize and pos[1] < cornerSize:
            pos += [0, cornerSize]
        elif pos[0] > self.config.SizeX - cornerSize and pos[1] > self.config.SizeY - cornerSize:
            pos -= [0, cornerSize]
        elif pos[0] > self.config.SizeX - cornerSize and pos[1] < cornerSize:
            pos += [-cornerSize, 0]
        elif pos[0] < cornerSize and pos[1] > self.config.SizeY - cornerSize:
            pos += [0, -cornerSize]

        return pos

    def computePointBoundingBox(self, fwhm):
        sigma = self.sigma(fwhm)
        lim = np.ceil(5 * sigma)

        return self.createBoundingBoxBySize(lim, lim)

    def createBoundingBoxByDistance(self, distanceX, distanceY) -> BoundingBox:
        minX = 0 if distanceX > 0 else -distanceX
        maxX = self.config.SizeX - distanceX if distanceX > 0 else self.config.SizeX

        minY = 0 if distanceY > 0 else -distanceY
        maxY = self.config.SizeY - distanceY if distanceY > 0 else self.config.SizeY

        return BoundingBox(minX, maxX, minY, maxY)

    def createBoundingBoxForStreak(self, vectorA, vectorB):
        x1 = np.abs(vectorA[0])
        x2 = np.abs(vectorB[0])
        y1 = np.abs(vectorA[1])
        y2 = np.abs(vectorB[1])

        x = np.floor(x1 + x2)
        y = np.floor(y1 + y2)

        minX, minY = x, y
        maxX, maxY = self.config.SizeX - x, self.config.SizeY - y

        return BoundingBox(minX, maxX, minY, maxY)

    def createBoundingBoxBySize(self, sizeX, sizeY):
        minX = sizeX
        maxX = self.config.SizeX - sizeX

        minY = sizeY
        maxY = self.config.SizeY - sizeY

        return BoundingBox(minX, maxX, minY, maxY)

    def isWithinImage(self, pos):
        return 0 <= pos[0] < self.config.SizeX and 0 <= pos[1] < self.config.SizeY

    def flip(self, pos):
        return pos[1], pos[0]

    def sigma(self, fwhm):
        return fwhm / 2.355

    def getHalfLengths(self, sigma, length):
        halfLengthA = length * sigma + 5 * sigma
        halfLengthB = 5 * sigma

        return halfLengthA, halfLengthB


'''
    reads FITS files which contain dark, bias, flat frames
'''


class FitsReader:
    config: configuration.Configuration

    def __init__(self, config):
        self.config = config

    def loadImage(self, filePath):
        fitsImage = fits.getdata(filePath, memmap=False)
        return fitsImage

    def loadAllImages(self, directory):
        fullFilePaths = glob.glob(f'{directory}/*.fits', recursive=False)
        images = []
        for filePath in fullFilePaths:
            images.append(self.loadImage(filePath))

        return images

    def loadBiasFrames(self):
        return self.loadAllImages(self.config.BiasFrame.dataDir)

    def loadDarkFrames(self):
        return self.loadAllImages(self.config.DarkFrame.dataDir)

    def loadFlatFrames(self):
        return self.loadAllImages(self.config.FlatFrame.dataDir)


'''
    reads files which contain real data 
'''


class FileReader:
    config: configuration.Configuration

    def __init__(self, config):
        self.config = config

    def getFilesInDir(self, directory, suffix):
        files = os.listdir(directory)

        filteredFiles = []
        for file in files:
            if file.endswith(suffix):
                filteredFiles.append(f'{directory}/{file}')

        return filteredFiles

    def readObjectFile(self, filename):
        file = open(filename, "r")

        header = []
        data = []
        for line in file:
            # to remove whitespace
            line = line.strip()

            # header of the table starts and ends with '--'
            if line.startswith('--'):
                # to remove '--' at the start and end of the line
                line = line.strip('-')
                header = line.split('\t')
                continue

            # after the header there is a table with object positions
            if len(header) > 0 and len(line) > 0:
                splitLine = line.split('\t')

                # some elements after splitting contain whitespaces
                strippedLine = [s.strip() for s in splitLine]

                # add stripped and split values to data
                data.append(strippedLine)

        df = pd.DataFrame(np.array(data), columns=header)

        # convert all columns to numeric
        columns = ['RA[deg]', 'DEC[deg]', 'X', 'Y', 'MJD', 'MAG', 'ERROR_MAG', 'ADU', 'ERROR_ADU', 'EXP_TIME']
        df[columns] = df[columns].apply(pd.to_numeric)

        return df

    def readStarsFile(self, filename):
        # read all data in TSV file
        tsvData = pd.read_csv(filename, sep='\t')

        # filter rows which are streaks (contains '|s' substring in column 'kurt')
        filteredData = tsvData.loc[tsvData['kurt'].str.endswith('|s'), :]

        # convert all columns to numeric
        columns = ['cent.x', 'cent.y', 'snr', 'iter', 'sum', 'mean', 'var', 'std', 'skew', 'bckg']
        filteredData[columns] = filteredData[columns].apply(pd.to_numeric)

        return filteredData

    def getFullStarFileName(self, directory, filename):
        index = filename.find('_a_m')
        # if we cant find substring it means that this name doesnt need trimming
        if index < 0:
            return f'{directory}\\{filename}.tsv'

        newFilename = filename[:index]
        return f'{directory}\\{newFilename}.tsv'

    def computeStarsFromFile(self, filename):
        stars = []

        data = self.readStarsFile(filename)
        for index, row in data.iterrows():
            star = self.starFromFile(row['cent.x'], row['cent.y'], row['sum'])
            stars.append(star)

        return stars

    def computeObjectsFromFile(self, filename):
        objects = []

        data = self.readObjectFile(filename)
        for index, row in data.iterrows():
            obj = self.objectFromFile(row['FILENAME'], row['X'], row['Y'], row['ADU'])
            objects.append(obj)

        return objects

    def starFromFile(self, x, y, brightness):
        fwhm = self.config.Stars.fwhm.value()
        length = self.config.Stars.length.value()
        alpha = self.config.Stars.alpha.value()

        return Star(x=x,
                    y=y,
                    brightness=brightness,
                    fwhm=fwhm,
                    length=length,
                    alpha=alpha)

    def objectFromFile(self, filename, x, y, brightness):
        fwhm = self.config.Objects.fwhm.value()

        obj = Object(x=x,
                     y=y,
                     brightness=brightness,
                     fwhm=fwhm,
                     length=0,
                     alpha=0,
                     positions=[[x, y]],
                     isCluster=False)

        objWithFilename = ObjectWithFile(obj, filename)

        return objWithFilename


'''
    draws gauss point and line
'''


class DrawingTool:

    def __init__(self, config):
        self.config = config
        self.utils = Utils(config)

    def computeLimits(self, x, y, limX, limY):
        upy = math.floor(max(0, y - limY))
        dwy = math.ceil(min(self.config.SizeY - 1, y + limY))

        upx = math.floor(max(0, x - limX))
        dwx = math.ceil(min(self.config.SizeX - 1, x + limX))

        return upy, dwy, upx, dwx

    def normalize(self, image):
        maxImg = np.max(image)
        imageNorm = (image / maxImg)

        return imageNorm

    def prepareImage(self, image, brightness, clip=True):
        imageNorm = self.normalize(image)
        imageBr = brightness * imageNorm

        # apply Poisson noise if enabled
        imagePoisson = self.poisson(imageBr)

        # clip values to interval 0,65535
        imageClipped = self.utils.clip(imagePoisson)

        return imageClipped if clip else imagePoisson

    def bicauchy(self, x, y, k, g2):
        return k / (x ** 2 + y ** 2 + g2) ** 1.5

    def bigaus(self, x, y, k, sigma):
        return k * np.exp(-0.5 * (x ** 2 + y ** 2) / sigma)

    def bigaus2(self, x, y, k, sigmaX, sigmaY, alpha):
        x2 = x ** 2
        y2 = y ** 2

        cos2 = np.cos(alpha) ** 2
        sin2 = np.sin(alpha) ** 2

        sin2x = np.sin(2 * alpha)

        a = 0.5 * ((cos2 / sigmaX) + (sin2 / sigmaY))
        b = 0.25 * (-(sin2x / sigmaX) + (sin2x / sigmaY))
        c = 0.5 * ((sin2 / sigmaX) + (cos2 / sigmaY))

        return k * np.exp(- (a * x2 + 2 * b * x * y + c * y2))

    def poisson(self, image):
        if self.config.applyPoisson:
            image = np.random.poisson(image)
        return image

    def _drawGalaxyCauchy(self, galaxy, image):
        gamma = galaxy.gamma

        k = gamma / 2 / np.pi
        gamma2 = gamma ** 2
        lim = 15 * gamma

        upy, dwy, upx, dwx = self.computeLimits(galaxy.x, galaxy.y, lim, lim)

        blankImage = np.zeros_like(image)
        for y in range(upy, dwy + 1):
            for x in range(upx, dwx + 1):
                value = self.bicauchy(galaxy.x - x + 0.5, galaxy.y - y + 0.5, k, gamma2)
                blankImage[y, x] += value

        newImage = self.prepareImage(blankImage, galaxy.brightness)

        return newImage

    def _drawBiGauss(self, object, image):

        alpha = object.alpha
        alphaRad = np.deg2rad(alpha)

        sigmaX = object.sigmaX
        sigmaY = object.sigmaY

        sigmaX2 = sigmaX ** 2
        sigmaY2 = sigmaY ** 2

        k = 1

        limX = np.ceil(15 * sigmaX)
        limY = np.ceil(15 * sigmaY)

        upy, dwy, upx, dwx = self.computeLimits(object.x, object.y, limX, limY)

        blankImage = np.zeros_like(image)
        for y in range(upy, dwy + 1):
            for x in range(upx, dwx + 1):
                value = self.bigaus2(object.x - x + 0.5, object.y - y + 0.5, k, sigmaX2, sigmaY2, alphaRad)
                blankImage[y, x] += value

        newImage = self.prepareImage(blankImage, object.brightness, clip=False)

        return newImage

    def drawGalaxyGaus(self, galaxy, image, dataList):

        sharpGalaxy = Galaxy(galaxy.x, galaxy.y, galaxy.brightness * (1 - galaxy.brightnessFactor), galaxy.alpha,
                             galaxy.sigmaFactor * galaxy.sigmaX,
                             galaxy.sigmaFactor * galaxy.sigmaY)
        diffuseGalaxy = Galaxy(galaxy.x, galaxy.y, galaxy.brightnessFactor * galaxy.brightness, galaxy.alpha,
                               galaxy.sigmaX, galaxy.sigmaY)

        sharpGalaxyImage = self._drawBiGauss(sharpGalaxy, image)
        diffuseGalaxyImage = self._drawBiGauss(diffuseGalaxy, image)

        newImage = diffuseGalaxyImage + sharpGalaxyImage

        newImage = self.utils.clip(newImage)

        # draw galaxy onto image
        image += newImage

        # add object to data list
        dataList.append(ObjectDataFactory('galaxy').fromImage(newImage, galaxy.x, galaxy.y))

    def drawStarGaus(self, star, image, dataList):

        sigma = self.utils.sigma(star.fwhm)
        sigma2 = sigma ** 2
        k = 1 / 2 / np.pi / sigma2
        lim = np.ceil(5 * sigma)

        upy, dwy, upx, dwx = self.computeLimits(star.x, star.y, lim, lim)

        blankImage = np.zeros_like(image)
        for y in range(upy, dwy + 1):
            for x in range(upx, dwx + 1):
                value = self.bigaus(star.x - x + 0.5, star.y - y + 0.5, k, sigma2)
                blankImage[y, x] += value

        newImage = self.prepareImage(blankImage, star.brightness)

        # draw point onto image
        image += newImage

        # add object to data list
        dataList.append(ObjectDataFactory('point').fromImage(newImage, star.x, star.y))

    def drawLineGauss(self, star, image, dataList):

        # center of object
        center = np.array([star.x, star.y])

        # streak variables
        length = star.length
        sigma = self.utils.sigma(star.fwhm)
        alpha = star.alpha
        alphaRad = np.deg2rad(alpha)

        # direction vector of the object (not normalized)
        shift = length * sigma * np.array([np.cos(alphaRad), np.sin(alphaRad)])

        # right and left point of the object (plateau boundaries)
        rightP = center + shift
        leftP = center - shift

        # playground corners (it is a rotated box)
        halfLengthA, halfLengthB = self.utils.getHalfLengths(sigma, length)

        corners = self.utils.getCorners(center, halfLengthA, halfLengthB, alphaRad)

        # rows/y coordinates (upy <= y <= dwy)
        upy = corners['bottom']
        dwy = corners['top']

        # cols/x coordinates (upx <= x <= dwx)
        upx = corners['left']
        dwx = corners['right']

        sigma2 = sigma ** 2
        alpha2 = alphaRad + np.pi / 2
        sin2 = np.sin(alpha2)
        cos2 = np.cos(alpha2)
        shift2 = np.array([cos2, sin2])

        # two points on the centre line
        c1 = center
        c2 = center + shift2

        # two points on the left line
        l1 = leftP
        l2 = leftP + shift2

        # two points on the right line
        r1 = rightP
        r2 = rightP + shift2

        blankImage = np.zeros_like(image)
        for y in range(upy, dwy + 1):
            for x in range(upx, dwx + 1):
                # left of left line
                if self.utils.checkLeft(x, y, l1, l2):
                    blankImage[y, x] += self.bigaus(leftP[0] - x + 0.5, leftP[1] - y + 0.5, 1, sigma2)
                # right of right line
                elif self.utils.checkRight(x, y, r1, r2):
                    blankImage[y, x] += self.bigaus(rightP[0] - x + 0.5, rightP[1] - y + 0.5, 1, sigma2)
                else:
                    # this is the strechted zone
                    # project point on the centre line (given by points centre and centre + direction_vec)
                    point = np.array([x, y])
                    projected = self.utils.projectPointOntoLine(point, c1, c2)
                    blankImage[y, x] += self.bigaus(center[0] - projected[0] + 0.5, center[1] - projected[1] + 0.5, 1,
                                                    sigma2)

        newImage = self.prepareImage(blankImage, star.brightness)

        # draw line onto image
        image += newImage

        # add object to data list
        dataList.append(ObjectDataFactory('line').fromImage(newImage, star.x, star.y))


'''
    draws noises and defects
'''


class DefectDrawingTool:
    config: configuration.Configuration
    utils: Utils
    fitsReader: FitsReader

    def __init__(self, config):
        self.config = config
        self.utils = Utils(config)
        self.fitsReader = FitsReader(config)

        self.neighbourhood4 = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        self.neighbourhood8 = self.neighbourhood4 + [[1, 1], [-1, -1], [1, -1], [-1, 1]]

        # load frames
        self.biasFrames = self.fitsReader.loadBiasFrames() if self.config.BiasFrame.enable else []
        self.darkFrames = self.fitsReader.loadDarkFrames() if self.config.DarkFrame.enable else []
        self.flatFrames = self.fitsReader.loadFlatFrames() if self.config.FlatFrame.enable else []

    '''
        this add all noises and defects that are different for each image 
        this includes: sky background noise, cosmics
    '''

    def addChangingDefects(self, image, dataList):

        # first add sky background noise
        self.addNoise(image)

        # second add cosmic rays which appear randomly at image
        self.addCosmicRays(image, dataList)

    '''
        this adds all defects and noises that need to be same for every frame in series
        this includes: flat, dark, bias frames, hot pixels, dead pixels, dead columns, traps
        the input is not one image but list of images
    '''

    def addFinalDefects(self, images, dataLists):

        # first add flat field frame which multiplies image by pixel sensitivity
        self.addFlat(images)

        # second add additive noises
        self.addBias(images)
        self.addDark(images)

        # third add noises/defects that set value not add
        # this includes: dead pixels, dead columns, traps
        self.addHotPixels(images, dataLists)

    def addNoise(self, image):
        if self.config.Noise.enable:
            noise_image = np.abs(
                self.config.Noise.std * np.random.randn(self.config.SizeY, self.config.SizeX) + self.config.Noise.mean)
            image += noise_image

    def addBias(self, images):
        if self.config.BiasFrame.enable:
            biasFrame = random.choice(self.biasFrames)

            for image in images:
                image += biasFrame

    def addDark(self, images):
        if self.config.DarkFrame.enable:
            darkFrame = random.choice(self.darkFrames)

            for image in images:
                image += darkFrame

    def addFlat(self, images):
        if self.config.FlatFrame.enable:
            flatFrame = random.choice(self.flatFrames)

            for image in images:
                image *= flatFrame


    def addHotPixels(self, images, dataLists):

        if self.config.HotPixel.enable:
            # We want the hot pixels to always be in the same places
            # (at least for the same image size) but also want them to appear to be randomly
            # distributed.
            count = self.config.HotPixel.count

            randomSeed = self.config.HotPixel.getRandomSeed()
            print(randomSeed)
            rng = np.random.RandomState(randomSeed)

            xList = rng.randint(0, self.config.SizeX - 1, size=count)
            yList = rng.randint(0, self.config.SizeY - 1, size=count)
            brightnessList = [self.config.HotPixel.brightness.value() for i in range(count)]

            for i in range(count):
                x = xList[i]
                y = yList[i]
                brightness = brightnessList[i]

                for i in range(len(images)):
                    image = images[i]
                    dataList = dataLists[i]

                    image[y, x] = brightness
                    dataList.append(ObjectDataFactory('hotpixel').fromData(x, y, brightness))

    def addCosmicRays(self, image, dataList):

        if self.config.CosmicRays.enable:
            count = self.config.CosmicRays.count.value()

            # sometimes the function creating cosmic doesnt add any cosmic
            # therefore instead of for cycle we count how many cosmics were actually added
            index = 0
            while index < count:
                cosmicType = random.choice(self.config.CosmicRays.cosmicTypes)
                #cosmicType = 'track'

                if cosmicType == 'spot':
                    index += self.addSpot(image, dataList)
                elif cosmicType == 'worm':
                    index += self.addWorm(image, dataList)
                elif cosmicType == 'track':
                    index += self.addTrack(image, dataList)

    def addSpot(self, image, dataList):
        newImage = np.zeros_like(image)
        pixelCount = self.config.CosmicRays.spotPixelCount.value()

        # compute center point
        lastPixel = self.utils.randomPosition(round=True)
        newImage[self.utils.flip(lastPixel)] = self.config.CosmicRays.brightness.value()

        outOfImageCount = 0
        outOfImagePatience = 5

        numberOfFilledPixels = 1
        while numberOfFilledPixels < pixelCount:
            direction = random.choice(self.neighbourhood4)
            position = lastPixel + direction

            # when we are out of image for too long the function stops
            # and returns 0 to signalize that no spot was added to image
            if outOfImageCount > outOfImagePatience:
                return 0

            # if position is not within image find a new direction
            if not self.utils.isWithinImage(position):
                print("skipping out of image")
                outOfImageCount += 1
                lastPixel = position
                continue

            # if position is already filled find a new direction
            if newImage[self.utils.flip(position)] > 0:
                print("skipping already filled")
                lastPixel = position
                continue

            newImage[self.utils.flip(position)] = self.config.CosmicRays.brightness.value()

            numberOfFilledPixels += 1
            # here we dont update the lastPixel because we want it centered around the startPoint
            # if we already filled all of them then we change the lastPixel

            # when pixel of the spot is added to image it means that we are now back in the image
            # therefore we need to clear out of image counter
            outOfImageCount = 0

        newImageClipped = self.utils.clip(newImage)

        image += newImageClipped

        dataList.append(ObjectDataFactory('cosmics').fromImage(newImageClipped))

        return 1

    def addWorm(self, image, dataList):
        newImage = np.zeros_like(image)
        pixelCount = self.config.CosmicRays.pixelCount.value()

        # compute start point and random direction
        lastPixel = self.utils.randomPosition(round=True)
        direction = random.choice(self.neighbourhood8)

        outOfImageCount = 0
        outOfImagePatience = 5

        numberOfFilledPixels = 0
        while numberOfFilledPixels < pixelCount:

            # compute new direction which is allowed
            direction = random.choice(self.allowedDirections(direction))

            # compute number of pixels for this direction
            directionPixelCount = random.randrange(1, (pixelCount // 4) + 1)

            for _ in range(directionPixelCount):
                position = lastPixel + direction

                # when we are out of image for too long the function stops
                # and returns 0 to signalize that no worm was added to image
                if outOfImageCount > outOfImagePatience:
                    return 0

                # if we have already enough filled pixels break the FOR loop
                if numberOfFilledPixels >= pixelCount:
                    break

                # if position is not within image break this for loop and find a new direction
                if not self.utils.isWithinImage(position):
                    print("skipping out of image")
                    outOfImageCount += 1
                    break

                # if position is already filled break this for loop and find new direction
                if newImage[self.utils.flip(position)] > 0:
                    print("skipping already filled")
                    break

                # compute brightness of this point
                newImage[self.utils.flip(position)] = self.config.CosmicRays.brightness.value()

                # when pixel of the spot is added to image it means that we are now back in the image
                # therefore we need to clear out of image counter
                outOfImageCount = 0

                # when finished update number of filled pixels and last pixel
                lastPixel = position
                numberOfFilledPixels += 1

        newImageClipped = self.utils.clip(newImage)

        image += newImageClipped

        dataList.append(ObjectDataFactory('cosmics').fromImage(newImageClipped))

        return 1

    def addTrack(self, image, dataList):
        newImage = np.zeros_like(image)
        pixelCount = self.config.CosmicRays.pixelCount.value()

        # compute start point and random directions
        boundingBox = self.utils.createBoundingBoxBySize(pixelCount / 3, pixelCount / 3)
        lastPixel = self.utils.randomPosition(round=True, boundingBox=boundingBox)
        mainDirection = random.choice(self.neighbourhood8)
        secondaryDirection = random.choice(self.allowedDirections(mainDirection))

        # boolean value to allow us to alternate between two directions
        useMainDirection = True

        # boolean value to indicate if we are out of image
        outOfImage = False

        numberOfFilledPixels = 0
        while numberOfFilledPixels < pixelCount:

            # compute number of pixels for this direction
            directionPixelCount = random.randrange(1, 5)

            for _ in range(directionPixelCount):
                # choose direction
                direction = mainDirection if useMainDirection else secondaryDirection

                # compute position
                position = lastPixel + direction

                # if we have already enough filled pixels break the FOR loop
                if numberOfFilledPixels >= pixelCount:
                    break

                # if position is not within image break this for loop and end while cycle
                if not self.utils.isWithinImage(position):
                    outOfImage = True
                    print("skipping out of image")
                    break

                # compute brightness of this point
                newImage[self.utils.flip(position)] = self.config.CosmicRays.brightness.value()

                # when finished update number of filled pixels and last pixel
                lastPixel = position
                numberOfFilledPixels += 1

            # when finished in certain direction we should alternate directions
            useMainDirection = not useMainDirection

            # if we are out of image we need to break out of while cycle as there is no more space where we can go
            if outOfImage:
                break

        # if we got out of image and filled less than defined number of pixels disregard this track and
        # return 0 to signalize that no track was added to image
        if outOfImage and numberOfFilledPixels < pixelCount:
            return 0

        newImageClipped = self.utils.clip(newImage)

        image += newImageClipped

        dataList.append(ObjectDataFactory('cosmics').fromImage(newImageClipped))

        return 1

    def allowedDirections(self, direction):
        x, y = direction
        if x == 0:
            return [[-1, y], [1, y]]

        if y == 0:
            return [[x, 1], [x, -1]]

        return [[x, 0], [0, y]]


'''
    saves data to TSV files
'''


class TSVSaver:
    config: configuration.Configuration

    def __init__(self, config):
        self.config = config

    def getDirectory(self, t):
        if self.config.oneFileOnly:
            return os.path.join(self.config.dataFile, 'TSV')
        return os.path.join(self.config.dataFile, f'tsv{t}')

    def savePositions(self, stars, galaxies, objects, t):
        directory = self.getDirectory(t)
        if not os.path.isdir(directory):
            os.mkdir(directory)

        for i in range(self.config.numberOfFramesInOneSeries):
            data = [s.toTSV() for s in stars] + [g.toTSV() for g in galaxies] + [o.toTSV(i) for o in objects]
            if len(data) > 0:
                df = pd.DataFrame(np.array(data), columns=["x", "y", "brightness", "is_object"])
                df.to_csv(f"{directory}/data_{t}_{i + 1:04d}.tsv", index=False, sep='\t')

        data = [[i] + o.toTSV(i) for o in objects for i in range(self.config.numberOfFramesInOneSeries)]
        if len(data) > 0:
            df = pd.DataFrame(np.array(data), columns=["image_number", "x", "y", "brightness", "is_object"])
            df.to_csv(f"{directory}/{t}_objects.tsv", index=False)

    def savePixelData(self, objectData, t):

        directory = self.getDirectory(t)
        if not os.path.isdir(directory):
            os.mkdir(directory)

        for i in range(self.config.numberOfFramesInOneSeries):
            data = [od.toTSV() for od in objectData[i]]
            if len(data) > 0:
                df = pd.DataFrame(np.array(data),
                                  columns=["object_type", "x", "y", "pixel_count", "total_intensity", "max_intensity"])
                df.to_csv(f"{directory}/pixel_data_{t}_{i + 1:04d}.tsv", index=False, sep='\t')


class ImageSaver:
    config: configuration.Configuration

    def __init__(self, config):
        self.config = config

    def getDirectory(self, t):
        if self.config.oneFileOnly:
            return os.path.join(self.config.dataFile, 'FITS')
        return os.path.join(self.config.dataFile, f'fits{t}')

    def saveFITSImages(self, images, t):
        directoryFITS = self.getDirectory(t)
        if not os.path.isdir(directoryFITS):
            os.mkdir(directoryFITS)

        for i in range(len(images)):
            self.saveImgToFits(images[i], f'{directoryFITS}/{t}_{i}')

    def savePNGImages(self, images, t):
        directoryPng = os.path.join(self.config.dataFile, 'PNG')
        if not os.path.isdir(directoryPng):
            os.mkdir(directoryPng)

        for i in range(len(images)):
            self.saveImgToPng(images[i], f'{directoryPng}/{t}_{i}')

    def saveImgToFits(self, image, name):
        name = f'{name}.fits'
        fits.writeto(name, image.astype(np.float32), overwrite=True)

    def saveImgToPng(self, image, name):
        name = f'{name}.png'
        cv2.imwrite(name, image)


class StarGenerator:
    config: configuration.Configuration
    fileReader: FileReader
    drawingTool: DrawingTool
    defectTool: DefectDrawingTool
    utils: Utils
    saverTSV: TSVSaver
    saverImage: ImageSaver

    def __init__(self, config, activateDefectTool=True):
        self.config: configuration.Configuration = config
        self.fileReader = FileReader(config)
        self.drawingTool = DrawingTool(config)
        self.defectTool = DefectDrawingTool(config) if activateDefectTool else None
        self.utils = Utils(config)
        self.saverTSV = TSVSaver(config)
        self.saverImage = ImageSaver(config)

    def generateOneSeriesFromFile(self, seriesNumber):

        t = str(int(time.time())) + f'_{config.index}_{seriesNumber}'

        # read files from directory
        directory = self.config.realData.file

        # there should be only one file for object positions
        objectFileName = self.fileReader.getFilesInDir(directory, '.txt')[0]

        # compute positions of object
        # the brightness of the object changes from frame to frame
        # therefore we cant represent it as one object with positions
        # instead its multiple objects, one object for one frame
        objectsWithFilenames = self.fileReader.computeObjectsFromFile(objectFileName)

        # create blank image
        blankImage = np.zeros((self.config.SizeY, self.config.SizeX))

        dataLists = []
        images = []

        for objWithFilename in objectsWithFilenames:

            # initialize empty data list
            dataList = []

            # initialize blank image
            image = blankImage.copy()

            # stars change in each frame
            # we need to read the corresponding file for this frame
            # each star will be drawn as a streak
            starFileName = self.fileReader.getFullStarFileName(directory, objWithFilename.filename)
            stars = self.fileReader.computeStarsFromFile(starFileName)
            for s in stars:
                self.drawObject(s, image, 'line', dataList)

            # there is only one object for each frame
            # object will be drawn as a point
            obj = objWithFilename.obj
            self.drawObject(obj, image, 'gauss', dataList)

            self.defectTool.addChangingDefects(image, dataList)

            image = self.utils.clip(image)

            images.append(image)
            dataLists.append(dataList)

        self.defectTool.addFinalDefects(images, dataLists)

        if self.config.savePixelData:
            self.saverTSV.savePixelData(dataLists, t)

        if self.config.saveFITSImages:
            self.saverImage.saveFITSImages(images, t)

        if self.config.savePNGImages:
            self.saverImage.savePNGImages(images, t)

        if self.config.plot:
            self.plotSeries(images)

    def generateSeries(self):
        for i in tqdm(range(self.config.numberOfSeries)):
            if self.config.realData.enabled:
                self.generateOneSeriesFromFile(i)
            else:
                self.generateOneSeries(i)

    def generateOneSeries(self, seriesNumber):

        # name of the file
        t = str(int(time.time())) + f'_{config.index}_{seriesNumber}'

        # TODO dat spat na normalne objekty
        objects = self.generateCleanObjects() + self.generateClusters()
        stars = self.getStars()
        galaxies = self.generateCleanGalaxies()

        # if we want to save positions of stars, galaxies and objects
        if self.config.savePositions:
            self.saverTSV.savePositions(stars, galaxies, objects, t)

        # generating images
        sameDataList = []
        images = []
        dataLists = []

        stars_image = np.zeros((self.config.SizeY, self.config.SizeX))

        for s in stars:
            self.drawObject(s, stars_image, self.config.Stars.method, sameDataList)

        for g in galaxies:
            self.drawingTool.drawGalaxyGaus(g, stars_image, sameDataList)

        for i in range(self.config.numberOfFramesInOneSeries):
            # initialize empty data list
            dataList = []

            # add objects to data list that are same for each photo
            dataList.extend(sameDataList)

            image = stars_image.copy()

            for obj in objects:
                method = self.config.Clusters.method if obj.isCluster else self.config.Objects.method

                self.drawObject(obj, image, method, dataList)

                obj.x, obj.y = (obj.positions[(i + 1) % self.config.numberOfFramesInOneSeries][0],
                                obj.positions[(i + 1) % self.config.numberOfFramesInOneSeries][1])

            # clip image
            image = self.utils.clip(image)

            self.defectTool.addChangingDefects(image, dataList)
            image = self.utils.clip(image)

            images.append(image)
            dataLists.append(dataList)

        self.defectTool.addFinalDefects(images, dataLists)

        if self.config.savePixelData:
            self.saverTSV.savePixelData(dataLists, t)

        if self.config.saveFITSImages:
            self.saverImage.saveFITSImages(images, t)

        if self.config.savePNGImages:
            self.saverImage.savePNGImages(images, t)

        if self.config.plot:
            self.plotSeries(images)

    def drawObject(self, obj, image, method, dataList):
        if method == 'line':
            return self.drawingTool.drawLineGauss(obj, image, dataList)
        elif method == 'gauss':
            return self.drawingTool.drawStarGaus(obj, image, dataList)

    def plotSeries(self, images):
        imagesCount = len(images)
        numberOfRows = imagesCount // 4
        if imagesCount % 4 != 0:
            numberOfRows += 1

        fig, axs = plt.subplots(numberOfRows, 4)
        for r in range(numberOfRows):
            for c in range(4):
                index = 4 * r + c
                if index < len(images):
                    self.utils.axis(r, c, axs).imshow(images[index], cmap='gray', vmin=0, vmax=50)
                    self.utils.axis(r, c, axs).set_title(f'image {index}')
                else:
                    self.utils.axis(r, c, axs).set_axis_off()
        plt.show()

    def getStars(self):
        if self.config.Stars.realData.enabled:
            return self.fileReader.computeStarsFromFile(self.config.Stars.realData.file)
        else:
            return self.generateStars()

    def generateStars(self):
        stars = [self.randomStar() for i in range(self.config.Stars.count.value())]
        return stars

    def generateObjects(self):
        objects = []
        if self.config.Objects.enable:
            objects = [self.randomObject() for i in range(self.config.Objects.count.value())]

        return objects

    def generateCleanObjects(self):
        objects = []
        if self.config.Objects.enable:
            objects = [self.randomCleanObject() for i in range(self.config.Objects.count.value())]

        return objects

    def generateCutObjects(self):
        objects = []
        if self.config.Objects.enable:
            objects = [self.randomCutObject() for i in range(self.config.Objects.count.value())]

        return objects

    def generateClusters(self):
        clusters = []
        if self.config.Clusters.enable:
            for i in range(self.config.Clusters.count.value()):
                oneCluster = self.generateOneCluster()
                clusters.extend(oneCluster)

        return clusters

    def generateOneCluster(self):
        # in the cluster all objects have same speed,length,alpha

        length = self.config.Clusters.length.value()
        alpha = self.config.Clusters.alpha.value()
        speed = self.config.Clusters.speed.value() / 100

        clusterObjects = []
        for i in range(self.config.Clusters.objectCountPerCluster.value()):
            obj = self.clusterObject(length, alpha, speed)
            clusterObjects.append(obj)

        return clusterObjects

    def generateGalaxies(self):
        galaxies = []
        if self.config.Galaxies.enable:
            galaxies = [self.randomGalaxy() for i in range(self.config.Galaxies.count.value())]

        return galaxies

    def generateCleanGalaxies(self):
        galaxies = []
        if self.config.Galaxies.enable:
            galaxies = [self.randomCleanGalaxy() for i in range(self.config.Galaxies.count.value())]

        return galaxies

    def randomStar(self):
        x, y = self.utils.randomPosition()
        brightness = self.config.Stars.brightness.value()
        fwhm = self.config.Stars.fwhm.value()
        length = self.config.Stars.length.value()
        alpha = self.config.Stars.alpha.value()

        return Star(x=x,
                    y=y,
                    brightness=brightness,
                    fwhm=fwhm,
                    length=length,
                    alpha=alpha)

    def randomCutObject(self):
        # this object is only for the purposes of training NN for cut lines
        # therefore object will stay in the same place through the whole series

        brightness = self.config.Objects.brightness.value()
        fwhm = self.config.Objects.fwhm.value()
        length = self.config.Objects.length.value()
        alpha = self.config.Objects.alpha.value()

        reversedBoundingBox = None
        if self.config.Objects.method == 'line':
            reversedBoundingBox = self.utils.computeReversedStreakBoundingBox(fwhm, length, alpha)

        startPoint = self.utils.randomPosition(boundingBox=reversedBoundingBox)
        # startPoint = self.utils.movePositionIfInCorner(startPoint)

        points = [startPoint for i in range(self.config.numberOfFramesInOneSeries)]

        return Object(x=startPoint[0],
                      y=startPoint[1],
                      brightness=brightness,
                      fwhm=fwhm,
                      length=length,
                      alpha=alpha,
                      positions=points,
                      isCluster=False)

    def randomCleanObject(self):
        brightness = self.config.Objects.brightness.value()
        fwhm = self.config.Objects.fwhm.value()
        length = self.config.Objects.length.value()
        alpha = self.config.Objects.alpha.value()

        speed = self.config.Objects.speed.value() / 100

        boundingBox = None
        if self.config.Objects.method == 'line':
            boundingBox = self.utils.computeStreakBoundingBox(fwhm, length, alpha)
        elif self.config.Objects.method == 'gauss':
            boundingBox = self.utils.computePointBoundingBox(fwhm)

        points = self.generateObjectPoints(alpha, speed, boundingBox)
        return Object(x=points[0][0],
                      y=points[0][1],
                      brightness=brightness,
                      fwhm=fwhm,
                      length=length,
                      alpha=alpha,
                      positions=points,
                      isCluster=False)

    def randomObject(self):
        brightness = self.config.Objects.brightness.value()
        fwhm = self.config.Objects.fwhm.value()
        length = self.config.Objects.length.value()
        alpha = self.config.Objects.alpha.value()

        speed = self.config.Objects.speed.value() / 100
        points = self.generateObjectPoints(alpha, speed)

        return Object(x=points[0][0],
                      y=points[0][1],
                      brightness=brightness,
                      fwhm=fwhm,
                      length=length,
                      alpha=alpha,
                      positions=points,
                      isCluster=False)

    def randomCleanGalaxy(self):
        brightness = self.config.Galaxies.brightness.value()
        alpha = self.config.Galaxies.alpha.value()
        sigmaX = self.config.Galaxies.sigmaX.value()
        sigmaY = self.config.Galaxies.sigmaY.value()
        brightnessFactor = self.config.Galaxies.brightnessFactor.value()
        sigmaFactor = self.config.Galaxies.sigmaFactor.value()

        boundingBox = self.utils.computeGalaxyBoundingBox(sigmaX, sigmaY)
        x, y = self.utils.randomPosition(boundingBox=boundingBox)

        return Galaxy(
            x=x,
            y=y,
            brightness=brightness,
            alpha=alpha,
            sigmaX=sigmaX,
            sigmaY=sigmaY,
            brightnessFactor=brightnessFactor,
            sigmaFactor=sigmaFactor
        )

    def randomGalaxy(self):
        x, y = self.utils.randomPosition()
        brightness = self.config.Galaxies.brightness.value()
        alpha = self.config.Galaxies.alpha.value()
        sigmaX = self.config.Galaxies.sigmaX.value()
        sigmaY = self.config.Galaxies.sigmaY.value()
        brightnessFactor = self.config.Galaxies.brightnessFactor.value()
        sigmaFactor = self.config.Galaxies.sigmaFactor.value()

        return Galaxy(
            x=x,
            y=y,
            brightness=brightness,
            alpha=alpha,
            sigmaX=sigmaX,
            sigmaY=sigmaY,
            brightnessFactor=brightnessFactor,
            sigmaFactor=sigmaFactor
        )

    def clusterObject(self, length, alpha, speed):
        brightness = self.config.Objects.brightness.value()
        fwhm = self.config.Objects.fwhm.value()
        points = self.generateObjectPoints(alpha, speed)

        return Object(x=points[0][0],
                      y=points[0][1],
                      brightness=brightness,
                      fwhm=fwhm,
                      length=length,
                      alpha=alpha,
                      positions=points,
                      isCluster=True)

    def generateObjectPoints(self, alpha, speed, cleanBoundingBox=None, limitPosition=True):
        # convert degrees to radians
        alphaRad = np.deg2rad(alpha)

        # compute shift vector from angle (normalized)
        shift = np.array([np.cos(alphaRad), np.sin(alphaRad)])

        minDimension = np.min([self.config.SizeY, self.config.SizeX])

        # compute total distance traveled
        totalDistX, totalDistY = (minDimension * speed * shift[0],
                                  minDimension * speed * shift[1])

        stepX, stepY = (totalDistX / self.config.numberOfFramesInOneSeries,
                        totalDistY / self.config.numberOfFramesInOneSeries)

        boundingBox = cleanBoundingBox
        if limitPosition:
            # create bounding box limiting object's position so it doesnt leave image
            positionBoundingBox = self.utils.createBoundingBoxByDistance(np.floor(totalDistX), np.floor(totalDistY))

            # merge position bounding box with clean bounding box if exists
            boundingBox = positionBoundingBox.merge(boundingBox, self.config.SizeX, self.config.SizeY)

        startPoint = self.utils.randomPosition(boundingBox=boundingBox)
        points = [(startPoint[0] + i * stepX, startPoint[1] + i * stepY) for i in
                  range(self.config.numberOfFramesInOneSeries)]

        return points


def generatePoints(config):
    darkdir = config.DarkFrame.dataDir
    for string in ['\\5s', '\\60s', '\\90s', '\\360s']:
        config.DarkFrame.dataDir = darkdir + string
        gen = StarGenerator(config)
        gen.generateSeries()


def generateGalaxies(config):
    configIndex = 0
    darkdir = config.DarkFrame.dataDir
    for string in ['\\5s', '\\60s', '\\90s', '\\360s']:
        config.DarkFrame.dataDir = darkdir + string

        defectDrawingTool = DefectDrawingTool(config)
        for i in np.arange(1.5, 4.5, 0.1):
            for j in np.arange(1.5, 4.5, 0.1):
                configIndex += 1

                config.index = configIndex
                config.Galaxies.sigmaX.fixedValue = i
                config.Galaxies.sigmaY.fixedValue = j

                gen = StarGenerator(config, activateDefectTool=False)
                gen.defectTool = defectDrawingTool
                gen.generateSeries()


def generateLines(config):
    configIndex = 0
    darkdir = config.DarkFrame.dataDir
    for string in ['\\5s', '\\60s', '\\90s', '\\360s']:
        config.DarkFrame.dataDir = darkdir + string

        defectDrawingTool = DefectDrawingTool(config)
        for alpha0 in range(0, 180):
            for length0 in range(1, 11):
                configIndex += 1

                config.index = f'a{alpha0}l{length0}'
                config.Objects.alpha.fixedValue = alpha0
                config.Objects.length.fixedValue = length0

                gen = StarGenerator(config, activateDefectTool=False)
                gen.defectTool = defectDrawingTool
                gen.generateSeries()


if __name__ == "__main__":
    config = configuration.loadConfig()
    #generatePoints(config)
    gen = StarGenerator(config)
    gen.generateSeries()