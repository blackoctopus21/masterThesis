import glob
import os.path
from pathlib import Path
import numpy as np

from astropy.io import fits


class Cutter:
    def __init__(self, size):
        self.size = size

    def cut(self, dataDirectory, saveDirectory):
        data = self._loadImages(dataDirectory)
        for name, image in data.items():
            # cut images with window
            cutImages = self._cutImage(image)

            # save images
            fileDir = f'{saveDirectory}/{name}'
            self._saveImages(cutImages, fileDir)

    def _saveImages(self, images, fileDir):
        if not os.path.isdir(fileDir):
            os.mkdir(fileDir)

        for i, image in enumerate(images):
            self._saveImgToFits(image, f'{fileDir}/{i}')

    def _saveImgToFits(self, image, name):
        name = f'{name}.fits'
        fits.writeto(name, image.astype(np.float32), overwrite=True)
        print(f'saving image to {name}')

    def _cutImage(self, image):
        cutImages = []
        for y in range(0, image.shape[0] - self.size, self.size//2):
            for x in range(0, image.shape[1] - self.size, self.size//2):
                cutImage = image[y:y + self.size, x:x + self.size]
                cutImages.append(cutImage)

        return cutImages


    def _getFilenameFromPath(self, fullFilePath):
        path = Path(fullFilePath)
        filename = path.name

        return filename

    def _loadImages(self, dataDir):
        fullFilePaths = glob.glob(f'{dataDir}/*.fit*', recursive=False)
        data = dict()

        for filePath in fullFilePaths:
            image = self._loadImage(filePath)
            imageName = self._getFilenameFromPath(filePath)

            data[imageName] = image

        return data

    def _loadImage(self, filePath):
        fitsImage = fits.getdata(filePath, memmap=False)
        return fitsImage

if __name__ == "__main__":

    dataDir = 'C:/Users/Admin/OneDrive/SKOLA/DIPLOMOVKA/masterThesis/stargen/masterFrames'
    saveDir = 'C:/Users/Admin/OneDrive/SKOLA/DIPLOMOVKA/masterThesis/stargen/masterFramesCut'
    cutter = Cutter(50)
    cutter.cut(dataDir, saveDir)


