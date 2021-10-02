import math
from dataclasses import dataclass
import numpy as np
import pandas as pd
from astropy.io import fits
from random import randrange as rr
import random
import os
import time

import configuration
from tqdm import tqdm

import matplotlib.pyplot as plt


@dataclass
class BoundingBox:
    minX: int
    maxX: int
    minY: int
    maxY: int


@dataclass
class Star:
    x: int
    y: int
    brightness: int
    fwhm: int
    length: int
    alpha: int

    def toTSV(self):
        return [self.x, self.y, self.brightness, self.fwhm, 0]


@dataclass
class Object:
    x: int
    y: int
    brightness: int
    fwhm: int
    length: int
    alpha: int
    positions: list
    isCluster: bool

    def toTSV(self, pos):
        return [self.positions[pos][0], self.positions[pos][1], self.brightness, self.fwhm, 1]


@dataclass
class Cluster:
    objects: list


@dataclass
class ObjectWithFile:
    obj: Object
    filename: str


@dataclass
class TSVObject:
    pixelsX: list
    pixelsY: list
    type: str

    boundingBox: BoundingBox = None

    def boundingBox(self):
        if self.boundingBox is None:
            minX = np.amin(self.pixelsX)
            maxX = np.amax(self.pixelsX)
            minY = np.amin(self.pixelsY)
            maxY = np.amax(self.pixelsY)

            self.boundingBox = BoundingBox(minX, maxX, minY, maxY)

        return self.boundingBox


'''
    util class with helping functions
'''


class Utils:
    config: configuration.Configuration

    def __init__(self, config):
        self.config = config

    def axis(self, r, c, axis):
        if axis.ndim > 1:
            return axis[r, c]
        else:
            return axis[c]

    def randomPosition(self, boundingBox=None):
        if boundingBox is None:
            x = rr(self.config.SizeX)
            y = rr(self.config.SizeY)
        else:
            x = rr(boundingBox.minX, boundingBox.maxX)
            y = rr(boundingBox.minY, boundingBox.maxY)

        return np.array([x, y])

    def createBoundingBox(self, distanceX, distanceY) -> BoundingBox:
        minX = 0 if distanceX > 0 else -distanceX
        maxX = self.config.SizeX - distanceX if distanceX > 0 else self.config.SizeX

        minY = 0 if distanceY > 0 else -distanceY
        maxY = self.config.SizeY - distanceY if distanceY > 0 else self.config.SizeY

        return BoundingBox(minX, maxX, minY, maxY)

    def isWithinImage(self, pos):
        return 0 <= pos[0] < self.config.SizeX and 0 <= pos[1] < self.config.SizeY

    def flip(self, pos):
        return pos[1],pos[0]


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
        df[['RA[deg]', 'DEC[deg]', 'X', 'Y', 'MJD', 'MAG', 'ERROR_MAG', 'ADU', 'ERROR_ADU', 'EXP_TIME']] = df[
            ['RA[deg]', 'DEC[deg]', 'X', 'Y', 'MJD', 'MAG', 'ERROR_MAG', 'ADU', 'ERROR_ADU', 'EXP_TIME']].apply(
            pd.to_numeric)

        return df

    def readStarsFile(self, filename):
        # read all data in TSV file
        tsvData = pd.read_csv(filename, sep='\t')

        # filter rows which are streaks (contains '|s' substring in column 'kurt')
        filteredData = tsvData.loc[tsvData['kurt'].str.endswith('|s'), :]

        # convert all columns to numeric
        filteredData[['cent.x', 'cent.y', 'snr', 'iter', 'sum', 'mean', 'var', 'std', 'skew', 'bckg']] = filteredData[
            ['cent.x', 'cent.y', 'snr', 'iter', 'sum', 'mean', 'var', 'std', 'skew', 'bckg']].apply(pd.to_numeric)

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

    def bigaus(self, x, y, k, sigma):
        return k * np.exp(-0.5 * (x ** 2 + y ** 2) / sigma)

    def drawStarGaus(self, star, image):

        # for the purposes ob TSV object
        pixelsX = []
        pixelsY = []

        sigma = (star.fwhm / 2.355)
        sigma2 = sigma ** 2
        k = 1 / 2 / np.pi / sigma2
        lim = np.ceil(5 * sigma)

        upy = math.floor(max(0, star.y - lim))
        dwy = math.ceil(min(self.config.SizeY - 1, star.y + lim))

        upx = math.floor(max(0, star.x - lim))
        dwx = math.ceil(min(self.config.SizeX - 1, star.x + lim))

        for y in range(upy, dwy + 1):
            for x in range(upx, dwx):
                value = star.brightness * self.bigaus(star.x - x + 0.5, star.y - y + 0.5, k, sigma2)
                image[y, x] += value

                pixelsX.append(x)
                pixelsY.append(y)

        return TSVObject(pixelsX, pixelsY, 'point')

    def drawLineGauss(self, star, image):
        def calcPosition(x, y, linePointA, linePointB):
            x1 = linePointA[0]
            y1 = linePointA[1]
            x2 = linePointB[0]
            y2 = linePointB[1]
            return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)

        def checkRight(x, y, linePointA, linePointB):
            return calcPosition(x, y, linePointA, linePointB) >= 0

        def checkLeft(x, y, linePointA, linePointB):
            return calcPosition(x, y, linePointA, linePointB) <= 0

        def inRange(point, max0, min0=0):
            if point < min0:
                return min0
            elif point > max0:
                return max0
            return point

        def getCorners(center, halfLengthA, halfLengthB, alpha):

            # basic vectors of the box
            vectorCenter = center
            vectorA = np.array([halfLengthA * np.cos(alpha), halfLengthA * np.sin(alpha)])
            vectorB = np.array([halfLengthB * np.cos(alpha + np.pi / 2), halfLengthB * np.sin(alpha + np.pi / 2)])

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

            boxTop = int(inRange(boxTop, self.config.SizeY - 1))
            boxBottom = int(inRange(boxBottom, self.config.SizeY - 1))
            boxLeft = int(inRange(boxLeft, self.config.SizeX - 1))
            boxRight = int(inRange(boxRight, self.config.SizeX - 1))

            return {'top': boxTop, 'bottom': boxBottom, 'left': boxLeft, 'right': boxRight}

        def projectPointOntoLine(point, linePointA, linePointB):
            # project point P onto line given by points A and B
            AP = point - linePointA
            AB = linePointB - linePointA
            return linePointA + (np.dot(AP, AB) / np.dot(AB, AB)) * AB

        # for the purposes ob TSV object
        pixelsX = []
        pixelsY = []

        # center of object
        center = np.array([star.x, star.y])

        # streak variables
        length = star.length
        sigma = (star.fwhm / 2.355)
        alpha = star.alpha
        alphaRad = np.deg2rad(alpha)

        # direction vector of the object (not normalized)
        shift = length * sigma * np.array([np.cos(alphaRad), np.sin(alphaRad)])

        # right and left point of the object (plateau boundaries)
        rightP = center + shift
        leftP = center - shift

        # playground corners (it is a rotated box)
        halfLengthA = length * sigma + 5 * sigma
        halfLengthB = 5 * sigma

        corners = getCorners(center, halfLengthA, halfLengthB, alphaRad)

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

        newImage = np.zeros_like(image)
        for y in range(upy, dwy + 1):
            for x in range(upx, dwx+1):
                # left of left line
                if checkLeft(x, y, l1, l2):
                    newImage[y, x] += self.bigaus(leftP[0] - x + 0.5, leftP[1] - y + 0.5, 1, sigma2)
                # right of right line
                elif checkRight(x, y, r1, r2):
                    newImage[y, x] += self.bigaus(rightP[0] - x + 0.5, rightP[1] - y + 0.5, 1, sigma2)
                else:
                    # this is the strechted zone
                    # project point on the centre line (given by points centre and centre + direction_vec)
                    point = np.array([x, y])
                    projected = projectPointOntoLine(point, c1, c2)
                    newImage[y, x] += self.bigaus(center[0] - projected[0] + 0.5, center[1] - projected[1] + 0.5, 1,
                                                  sigma2)

                pixelsX.append(x)
                pixelsY.append(y)

        sumImg = np.sum(newImage)
        newImage = star.brightness * (newImage / sumImg)

        image += newImage

        return TSVObject(pixelsX, pixelsY, 'line')


'''
    draws noises and defects
'''


class DefectDrawingTool:
    config: configuration.Configuration
    utils: Utils

    def __init__(self, config):
        self.config = config
        self.utils = Utils(config)

        self.neighbourhood4 = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        self.neighbourhood8 = self.neighbourhood4 + [[1, 1], [-1, -1], [1, -1], [-1, 1]]

    '''
        this adds all noises and defects that are same for each image and can be added at the beginning
        this includes: bias, dark current, flat field, diffuse sources       
    '''

    def addInitialDefects(self, image):
        # TODO
        pass

    '''
        this add all noises and defects that are different for each image 
        this includes: photon noise, cosmics, readout
    '''

    def addChangingDefects(self, image):
        objectsTSV = []

        self.addNoise(image)
        objectsTSV.extend(self.addCosmicRays(image))

        return objectsTSV

    '''
        this adds all defects and noises that need to be same for every frame in series
        and cant be added before generating objects because they set value and not add it
        this includes: hot pixels, dead pixels, dead columns, traps

        the input is not one image but list of images
    '''

    def addFinalDefects(self, images):
        objectsTSV = []

        objectsTSV.extend(self.addHotPixels(images))

        return objectsTSV

    def addNoise(self, image):
        if self.config.Noise.enable:
            noise_image = np.abs(
                self.config.Noise.std * np.random.randn(self.config.SizeX, self.config.SizeY) + self.config.Noise.mean)
            image += noise_image

    def addBias(self, image):
        if self.config.Bias.enable:
            value = self.config.Bias.value
            numberOfColumns = self.config.Bias.columns
            shape = image.shape

            bias_image = np.zeros_like(image) + value

            # We want a random-looking variation in the bias, but unlike the readnoise the bias should
            # *not* change from image to image, so we make sure to always generate the same "random" numbers.
            rng = np.random.RandomState(seed=8392)
            columns = rng.randint(0, self.config.SizeY, size=numberOfColumns)

            # This adds a little random-looking noise into the data.
            columnPattern = rng.randint(0, value, size=self.config.SizeX)

            # Make the chosen columns a little brighter than the rest
            for c in columns:
                bias_image[:, c] = value + columnPattern

            image += bias_image

    def addHotPixels(self, images):
        objectsTSV = []

        if self.config.HotPixel.enable:
            # We want the hot pixels to always be in the same places
            # (at least for the same image size) but also want them to appear to be randomly
            # distributed. Random seed will be dependent on the image size
            count = self.config.HotPixel.count

            randomSeed = self.config.SizeX * self.config.SizeY
            rng = np.random.RandomState(randomSeed)

            x = rng.randint(0, self.config.SizeX - 1, size=count)
            y = rng.randint(0, self.config.SizeY - 1, size=count)

            for i in range(count):
                objectsTSV.append(TSVObject([x[i]], [y[i]], 'hp'))
                for image in images:
                    image[tuple([y[i], x[i]])] = self.config.HotPixel.brightness.value()

        return objectsTSV

    def addCosmicRays(self, image):
        objectsTSV = []

        if self.config.CosmicRays.enable:
            count = self.config.CosmicRays.count.value()

            for _ in range(count):
                cosmicType = random.choice(self.config.CosmicRays.cosmicTypes)

                if cosmicType == 'spot':
                    objectsTSV.append(self.addSpot(image))
                elif cosmicType == 'worm':
                    objectsTSV.append(self.addWorm(image))
                elif cosmicType == 'track':
                    objectsTSV.append(self.addTrack(image))

        return objectsTSV

    def addSpot(self, image):
        newImage = np.zeros_like(image)
        pixelCount = self.config.CosmicRays.spotPixelCount.value()

        # for the purposes of TSV Object
        pixelsX = []
        pixelsY = []

        # compute center point
        lastPixel = self.utils.randomPosition()
        image[lastPixel[::-1]] = self.config.CosmicRays.brightness.value()

        numberOfFilledPixels = 1
        while numberOfFilledPixels < pixelCount:
            direction = random.choice(self.neighbourhood4)
            position = lastPixel + direction

            # if position is not within image find a new direction
            if not self.utils.isWithinImage(position):
                print("skipping out of image")
                lastPixel = position
                continue

            # if position is already filled find a new direction
            if newImage[self.utils.flip(position)] > 0:
                print("skipping already filled")
                lastPixel = position
                continue

            image[self.utils.flip(position)] = self.config.CosmicRays.brightness.value()
            pixelsX.append(position[0])
            pixelsY.append(position[1])

            numberOfFilledPixels += 1
            # here we dont update the lastPixel because we want it centered around the startPoint
            # if we already filled all of them then we change the lastPixel

        image += newImage

        return TSVObject(pixelsX, pixelsY, 'cosmic')

    def addWorm(self, image):
        newImage = np.zeros_like(image)
        pixelCount = self.config.CosmicRays.pixelCount.value()

        # for the purposes of TSV Object
        pixelsX = []
        pixelsY = []

        # compute start point and random direction
        lastPixel = self.utils.randomPosition()
        direction = random.choice(self.neighbourhood8)

        numberOfFilledPixels = 0
        while numberOfFilledPixels < pixelCount:

            # compute new direction which is allowed
            direction = random.choice(self.allowedDirections(direction))

            # compute number of pixels for this direction
            directionPixelCount = random.randrange(1, (pixelCount // 4) + 1)

            for _ in range(directionPixelCount):
                position = lastPixel + direction

                # if we have already enough filled pixels break the FOR loop
                if numberOfFilledPixels >= pixelCount:
                    break

                # if position is not within image break this for loop and find a new direction
                if not self.utils.isWithinImage(position):
                    print("skipping out of image")
                    break

                # if position is already filled break this for loop and find new direction
                if newImage[self.utils.flip(position)] > 0:
                    print("skipping already filled")
                    break

                # compute brightness of this point
                newImage[self.utils.flip(position)] = self.config.CosmicRays.brightness.value()
                pixelsX.append(position[0])
                pixelsY.append(position[1])

                # when finished update number of filled pixels and last pixel
                lastPixel = position
                numberOfFilledPixels += 1

        image += newImage

        return TSVObject(pixelsX, pixelsY, 'cosmic')

    def addTrack(self, image):
        newImage = np.zeros_like(image)
        pixelCount = self.config.CosmicRays.pixelCount.value()

        # for the purposes of TSV Object
        pixelsX = []
        pixelsY = []

        # compute start point and random directions
        lastPixel = self.utils.randomPosition()
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
                pixelsX.append(position[0])
                pixelsY.append(position[1])

                # when finished update number of filled pixels and last pixel
                lastPixel = position
                numberOfFilledPixels += 1

            # when finished in certain direction we should alternate directions
            useMainDirection = not useMainDirection

            # if we are out of image we need to break out of while cycle as there is no more space where we can go
            if outOfImage:
                break

        image += newImage

        return TSVObject(pixelsX, pixelsY, 'cosmic')

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



class StarGenerator:
    config: configuration.Configuration
    fileReader: FileReader
    drawingTool: DrawingTool
    defectTool: DefectDrawingTool
    utils: Utils

    def __init__(self, config):
        self.config: configuration.Configuration = config
        self.fileReader = FileReader(config)
        self.drawingTool = DrawingTool(config)
        self.defectTool = DefectDrawingTool(config)
        self.utils = Utils(config)

    def generateOneSeriesFromFile(self):

        t = int(time.time())

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

        self.defectTool.addInitialDefects(blankImage)

        allObjectsTSV = []
        images = []
        for objWithFilename in objectsWithFilenames:
            objectsTSV = []

            image = blankImage.copy()

            # stars change in each frame
            # we need to read the corresponding file for this frame
            # each star will be drawn as a streak
            starFileName = self.fileReader.getFullStarFileName(directory, objWithFilename.filename)
            stars = self.fileReader.computeStarsFromFile(starFileName)
            for s in stars:
                objectsTSV.append(self.drawObject(s, image, 'line'))

            # there is only one object for each frame
            # object will be drawn as a point
            obj = objWithFilename.obj
            objectsTSV.append(self.drawObject(obj, image, 'gauss'))

            objectsTSV.extend(self.defectTool.addChangingDefects(image))

            images.append(image)
            allObjectsTSV.append(objectsTSV)

        sameObjectsTSV = self.defectTool.addFinalDefects(images)

        # TODO here I need to save TSV objects somehow

        objects = [objFn.obj for objFn in objectsWithFilenames]
        self.saveSeriesToFile(images, objects, t)

        if self.config.plot:
            self.plotSeries(images)

    def generateSeries(self):
        for _ in tqdm(range(self.config.numberOfSeries)):
            if self.config.realData.enabled:
                self.generateOneSeriesFromFile()
            else:
                self.generateOneSeries()

    def generateOneSeries(self):

        t = int(time.time())

        objects = self.generateObjects()
        stars = self.getStars()
        clusters = self.generateClusters()

        # converge all objects together
        allObjects = objects
        for cl in clusters:
            allObjects.extend(cl.objects)

        self.saveTSV(stars, allObjects, t)

        if self.config.saveImages:
            sameObjectsTSV = []

            stars_image = np.zeros((self.config.SizeY, self.config.SizeX))
            for s in stars:
                sameObjectsTSV.append(self.drawObject(s, stars_image, self.config.Stars.method))

            self.defectTool.addInitialDefects(stars_image)

            images = []
            allObjectsTSV = []
            for i in range(self.config.numberOfFramesInOneSeries):
                objectsTSV = []

                image = stars_image.copy()

                objectsTSV.extend(self.defectTool.addChangingDefects(image))

                for obj in allObjects:
                    method = self.config.Clusters.method if obj.isCluster else self.config.Objects.method

                    objectsTSV.append(self.drawObject(obj, image, method))

                    obj.x, obj.y = (obj.positions[(i + 1) % self.config.numberOfFramesInOneSeries][0],
                                    obj.positions[(i + 1) % self.config.numberOfFramesInOneSeries][1])

                images.append(image)
                allObjectsTSV.append(objectsTSV)

            sameObjectsTSV.extend(self.defectTool.addFinalDefects(images))

            # TODO here I need to save TSV objects somehow

            self.saveSeriesToFile(images, allObjects, t)

            if self.config.plot:
                self.plotSeries(images)

    def drawObject(self, obj, image, method):
        if method == 'line':
            return self.drawingTool.drawLineGauss(obj, image)
        elif method == 'gauss':
            return self.drawingTool.drawStarGaus(obj, image)

    def saveTSV(self, stars, objects, t):

        directory = os.path.join(self.config.dataFile, f'tsv{t+random.random()}')
        os.mkdir(directory)

        for i in range(self.config.numberOfFramesInOneSeries):
            data = [s.toTSV() for s in stars] + [o.toTSV(i) for o in objects]
            if len(data) > 0:
                df = pd.DataFrame(np.array(data), columns=["x", "y", "brightness", "fwhm", "is_object"])
                df.to_csv(f"{directory}/data_{i + 1:04d}.tsv", index=False, sep='\t')

        data = [[i] + o.toTSV(i) for o in objects for i in range(self.config.numberOfFramesInOneSeries)]
        if len(data) > 0:
            df = pd.DataFrame(np.array(data), columns=["image_number", "x", "y", "brightness", "fwhm", "is_object"])
            df.to_csv(f"{directory}/objects.tsv", index=False)

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

    def saveSeriesToFile(self, images, objects, t):
        directory = os.path.join(self.config.dataFile, f'fits{t+random.random()}')
        os.mkdir(directory)
        for i in range(len(images)):
            name = f'{directory}/{i}'
            self.saveImgToFits(images[i], name)

        with open(f'{directory}/objects.txt', 'w') as f:
            for obj in objects:
                print(' '.join(list(map(str, obj.positions))), file=f)

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

    def generateClusters(self):
        clusters = []
        if self.config.Clusters.enable:
            clusters = [self.generateOneCluster() for i in range(self.config.Clusters.count.value())]

        return clusters

    def randomStar(self):
        x = rr(self.config.SizeX)
        y = rr(self.config.SizeY)
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

    def randomObject(self):
        brightness = self.config.Objects.brightness.value()
        fwhm = self.config.Objects.fwhm.value()
        length = self.config.Objects.length.value()
        alpha = self.config.Objects.alpha.value()

        speed = self.config.Objects.speed.value() / 100
        points = self.generateObjectPoints(alpha, speed)

        obj = Object(x=points[0][0],
                     y=points[0][1],
                     brightness=brightness,
                     fwhm=fwhm,
                     length=length,
                     alpha=alpha,
                     positions=points,
                     isCluster=False)

        return obj

    def clusterObject(self, length, alpha, speed):
        brightness = self.config.Objects.brightness.value()
        fwhm = self.config.Objects.fwhm.value()
        points = self.generateObjectPoints(alpha, speed)

        obj = Object(x=points[0][0],
                     y=points[0][1],
                     brightness=brightness,
                     fwhm=fwhm,
                     length=length,
                     alpha=alpha,
                     positions=points,
                     isCluster=True)

        return obj

    def generateObjectPoints(self, alpha, speed):
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

        # create bounding box limiting object's position so it doesnt leave image
        boundingBox = self.utils.createBoundingBox(np.floor(totalDistX), np.floor(totalDistY))

        startPoint = self.utils.randomPosition(boundingBox)
        points = [(startPoint[0] + i * stepX, startPoint[1] + i * stepY) for i in
                  range(self.config.numberOfFramesInOneSeries)]

        return points

    def generateOneCluster(self):
        # in the cluster all objects have same speed,length,alpha

        length = self.config.Clusters.length.value()
        alpha = self.config.Clusters.alpha.value()
        speed = self.config.Clusters.speed.value() / 100

        clusterObjects = []
        for i in range(self.config.Clusters.objectCountPerCluster.value()):
            obj = self.clusterObject(length, alpha, speed)
            clusterObjects.append(obj)

        return Cluster(clusterObjects)

    def saveImgToFits(self, image, name):
        name = f'{name}.fits'
        fits.writeto(name, image.astype(np.float32), overwrite=True)


class TrainingDataGenerator(StarGenerator):

    def generateData(self, N, k=3):

        half_examples = N // 2

        data_X = []
        data_y = []

        for _ in range(half_examples):
            res = []
            for _ in range(k):
                res += self.randomStar().toTSV()[:-1]

            data_X.append(res)
            data_y.append(0)

        for _ in range(half_examples):
            obj = self.randomObject()
            i = random.randrange(0, 8 - k + 1)
            res = []

            for j in range(k):
                res += obj.toTSV(i + j)[:-1]

            data_X.append(res)
            data_y.append(1)

        data_X = np.array(data_X)
        data_y = np.array(data_y)

        idx = np.arange(0, len(data_y))
        np.random.shuffle(idx)

        data_X = data_X[idx]
        data_y = data_y[idx]

        return data_X, data_y


if __name__ == "__main__":
    config = configuration.loadConfig()

    gen = StarGenerator(config)
    gen.generateSeries()

    # gen = TrainingDataGenerator(config)

    # res = gen.generateData(10)

    # print(res)
