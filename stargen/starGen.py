from dataclasses import dataclass

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from collections import namedtuple
from random import randrange as rr
import pathlib
import random
import os
import time
import configuration
from tqdm import tqdm


@dataclass
class Star:
    x: int
    y: int
    brightness: int
    fwhm: int

    def toTSV(self):
        return [self.x, self.y, self.brightness, self.fwhm, 0]


@dataclass
class Object:
    x: int
    y: int
    brightness: int
    fwhm: int
    positions: list

    def toTSV(self, pos):
        return [self.positions[pos][0], self.positions[pos][1], self.brightness, self.fwhm, 1]


class StarGenerator:

    def __init__(self, config):
        self.config: configuration.Configuration = config

    def generateSeries(self):
        for i in tqdm(range(self.config.numberOfSeries)):
            self.generateOneSeries()

    def generateOneSeries(self):

        t = int(time.time())

        objects = [self.randomObject() for i in range(self.config.Objects.count.value())]
        stars = self.generateStars()

        self.saveTSV(stars, objects, t)

        if self.config.saveImages:

            stars_image = np.zeros((self.config.SizeX, self.config.SizeY))
            for s in stars:
                if self.config.Stars.streak.enable:
                    self.drawLineGauss(s, stars_image)
                else:
                    self.drawStarGaus(s, stars_image)

            images = []
            for i in range(self.config.numberOfFramesInOneSeries):
                image = stars_image.copy()
                self.addNoise(image)

                for obj in objects:
                    self.drawStarGaus(obj, image)
                    obj.x, obj.y = obj.positions[(i + 1) % self.config.numberOfFramesInOneSeries][0], \
                                   obj.positions[(i + 1) % self.config.numberOfFramesInOneSeries][1]

                images.append(image)

            self.saveSeriesToFile(images, objects, t)

            if self.config.plot:
                self.plotSeries(images)

    def saveTSV(self, stars, objects, t):

        directory = os.path.join(self.config.dataFile, f'tsv{t}')
        os.mkdir(directory)

        for i in range(self.config.numberOfFramesInOneSeries):
            data = [s.toTSV() for s in stars] + [o.toTSV(i) for o in objects]
            df = pd.DataFrame(np.array(data), columns=["x", "y", "brightness", "fwhm", "is_object"])
            df.to_csv(f"{directory}/data_{i + 1:04d}.tsv", index=False, sep='\t')

        data = [[i] + o.toTSV(i) for o in objects for i in range(self.config.numberOfFramesInOneSeries)]
        df = pd.DataFrame(np.array(data), columns=["image_number", "x", "y", "brightness", "fwhm", "is_object"])
        df.to_csv(f"{directory}/objects.tsv", index=False)

    def plotSeries(self, images):
        numberOfRows = self.config.numberOfFramesInOneSeries // 4
        if self.config.numberOfFramesInOneSeries % 4 != 0:
            numberOfRows += 1

        fig, axs = plt.subplots(numberOfRows, 4)
        for r in range(numberOfRows):
            for c in range(4):
                index = 4 * r + c
                if index < len(images):
                    self.axis(r, c, axs).imshow(images[index], cmap='gray', vmin=0, vmax=50)
                    self.axis(r, c, axs).set_title(f'image {index}')
                else:
                    self.axis(r, c, axs).set_axis_off()
        plt.show()

    def axis(self, r, c, axis):
        if axis.ndim > 1:
            return axis[r, c]
        else:
            return axis[c]

    def saveSeriesToFile(self, images, objects, t):
        directory = os.path.join(self.config.dataFile, f'fits{t}')
        os.mkdir(directory)
        for i in range(self.config.numberOfFramesInOneSeries):
            name = f'{directory}/{i}'
            self.saveImgToFits(images[i], name)

        with open(f'{directory}/objects.txt', 'w') as f:
            for obj in objects:
                print(' '.join(list(map(str, obj.positions))), file=f)

    def generateStars(self):

        stars = [self.randomStar() for i in range(self.config.Stars.count.value())]

        return stars

    def randomStar(self):
        x = rr(self.config.SizeX)
        y = rr(self.config.SizeY)
        brightness = self.config.Stars.brightness.value()
        fwhm = self.config.Stars.fwhm.value()

        return Star(x, y, brightness, fwhm)

    def randomObject(self):
        points = self.generateRandomObjectPoints()
        brightness = self.config.Objects.brightness.value()
        fwhm = self.config.Objects.fwhm.value()

        obj = Object(x=points[0][0], y=points[0][1],
                     brightness=brightness, fwhm=fwhm, positions=points)

        return obj

    def generateRandomObjectPoints(self):
        x = rr(self.config.SizeX)
        y = rr(self.config.SizeY)
        p = random.random()
        if p >= 0.75:
            edge_point = (rr(self.config.SizeX), 0)
        elif p >= 0.5:
            edge_point = (rr(self.config.SizeX), 1024)
        elif p >= 0.25:
            edge_point = (0, rr(self.config.SizeY))
        else:
            edge_point = (1024, rr(self.config.SizeY))
        speed = self.config.Objects.speed.value() / 100

        step_x, step_y = ((edge_point[0] - x) * speed // self.config.numberOfFramesInOneSeries,
                          (edge_point[1] - y) * speed // self.config.numberOfFramesInOneSeries)
        points = [(x + i * step_x, y + i * step_y) for i in range(self.config.numberOfFramesInOneSeries)]

        return points

    def drawStarGaus(self, star, image):

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
                image[x, y] += star.brightness * self.bigaus(star.x - x + 0.5, star.y - y + 0.5, k, sigma2)

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

        # center of object
        center = np.array([star.x, star.y])

        # streak variables
        length = self.config.Stars.streak.length
        sigma = self.config.Stars.streak.sigma
        alpha = self.config.Stars.streak.alpha

        # direction vector of the object (not normalized)
        shift = length * sigma * np.array([np.cos(alpha), np.sin(alpha)])

        # right and left point of the object (plateau boundaries)
        rightP = center + shift
        leftP = center - shift

        # playground corners (it is a rotated box)
        halfLengthA = length * sigma + 5 * sigma
        halfLengthB = 5 * sigma

        corners = getCorners(center, halfLengthA, halfLengthB, alpha)

        # rows/y coordinates (upy <= y <= dwy)
        upy = corners['bottom']
        dwy = corners['top']

        # cols/x coordinates (upx <= x <= dwx)
        upx = corners['left']
        dwx = corners['right']

        sigma2 = sigma ** 2
        alpha2 = alpha + np.pi / 2
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
            for x in range(upx, dwx):
                # left of left line
                if checkLeft(x, y, l1, l2):
                    newImage[x, y] += self.bigaus(leftP[0] - x + 0.5, leftP[1] - y + 0.5, 1, sigma2)
                # right of right line
                elif checkRight(x, y, r1, r2):
                    newImage[x, y] += self.bigaus(rightP[0] - x + 0.5, rightP[1] - y + 0.5, 1, sigma2)
                else:
                    # this is the strechted zone
                    # project point on the centre line (given by points centre and centre + direction_vec)
                    point = np.array([x, y])
                    projected = projectPointOntoLine(point, c1, c2)
                    newImage[x, y] += self.bigaus(center[0] - projected[0] + 0.5, center[1] - projected[1] + 0.5, 1,
                                                  sigma2)

        newImage = star.brightness * (newImage / np.sum(newImage))

        image += newImage

    def bigaus(self, x, y, k, sigma):
        return k * np.exp(-0.5 * (x ** 2 + y ** 2) / sigma)

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
            columns = rng.randint(0, shape[1], size=numberOfColumns)

            # This adds a little random-looking noise into the data.
            columnPattern = rng.randint(0, value, size=shape[0])

            # Make the chosen columns a little brighter than the rest
            for c in columns:
                bias_image[:, c] = value + columnPattern

            image += bias_image

    def addHotPixels(self, image):
        if self.config.HotPixel.enable:
            # Like with the bias image, we want the hot pixels to always be in the same places
            # (at least for the same image size) but also want them to appear to be randomly
            # distributed. So we set a random number seed to ensure we always get the same thing.
            rng = np.random.RandomState(16201649)
            hotX = rng.randint(0, image.shape[1], size=self.config.HotPixel.count)
            hotY = rng.randint(0, image.shape[0], size=self.config.HotPixel.count)

            for i in range(self.config.HotPixel.count):
                image[tuple([hotY[i], hotX[i]])] = self.config.HotPixel.brightness.value()

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
