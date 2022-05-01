from abc import abstractmethod
from dataclasses import dataclass
import random
import yaml
from abc import ABC


@dataclass
class DataFile:
    enabled: bool
    file: str

    def __init__(self, **entries):
        self.__dict__.update(entries)


@dataclass
class Value(ABC):
    random: str
    savedValue = None
    fixedValue = None

    def __init__(self, **entries):
        self.__dict__.update(entries)

    def value(self):
        if self.fixedValue is not None:
            return self.fixedValue

        if self.random == 'uniform':
            return self.uniform()

        if self.random == 'normal':
            return self.normal()

        if self.random == 'uniform*':
            if self.savedValue is None:
                self.savedValue = self.uniform()
            return self.savedValue

        if self.random == 'normal*':
            if self.savedValue is None:
                self.savedValue = self.normal()
            return self.savedValue

    @abstractmethod
    def normal(self):
        ...

    @abstractmethod
    def uniform(self):
        ...


@dataclass
class FloatValue(Value):
    min: float
    max: float

    def normal(self):
        return random.gauss(mu=(abs(self.min + self.max) / 2), sigma=(self.max - self.min))

    def uniform(self):
        return random.uniform(self.min, self.max)


@dataclass
class IntValue(Value):
    min: int
    max: int

    def uniform(self):
        return random.randrange(self.min, self.max + 1)

    def normal(self):
        return int(random.gauss(mu=(abs(self.min + self.max) / 2), sigma=(self.max - self.min)))


@dataclass
class Noise:
    mean: int
    std: int
    enable: bool

    def __init__(self, **entries):
        self.__dict__.update(entries)


@dataclass
class Stars:
    count: IntValue
    brightness: FloatValue
    fwhm: FloatValue
    method: str
    length: IntValue
    alpha: IntValue
    realData: DataFile

    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.count = IntValue(**self.count)
        self.brightness = FloatValue(**self.brightness)
        self.fwhm = FloatValue(**self.fwhm)
        self.length = IntValue(**self.length)
        self.alpha = IntValue(**self.alpha)
        self.realData = DataFile(**self.realData)


@dataclass
class Objects:
    count: IntValue
    brightness: FloatValue
    fwhm: FloatValue
    speed: IntValue
    method: str
    length: IntValue
    alpha: IntValue
    enable: bool

    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.count = IntValue(**self.count)
        self.brightness = FloatValue(**self.brightness)
        self.fwhm = FloatValue(**self.fwhm)
        self.speed = IntValue(**self.speed)
        self.length = IntValue(**self.length)
        self.alpha = IntValue(**self.alpha)


@dataclass
class Clusters:
    count: IntValue
    objectCountPerCluster: IntValue
    brightness: FloatValue
    fwhm: FloatValue
    speed: IntValue
    method: str
    length: IntValue
    alpha: IntValue
    enable: bool

    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.count = IntValue(**self.count)
        self.objectCountPerCluster = IntValue(**self.objectCountPerCluster)
        self.brightness = FloatValue(**self.brightness)
        self.fwhm = FloatValue(**self.fwhm)
        self.speed = IntValue(**self.speed)
        self.length = IntValue(**self.length)
        self.alpha = IntValue(**self.alpha)


@dataclass
class Galaxies:
    enable: bool
    count: IntValue
    brightness: FloatValue
    alpha: IntValue
    sigmaX: FloatValue
    sigmaY: FloatValue
    brightnessFactor: FloatValue
    sigmaFactor: FloatValue

    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.count = IntValue(**self.count)
        self.brightness = FloatValue(**self.brightness)
        self.alpha = IntValue(**self.alpha)
        self.sigmaX = FloatValue(**self.sigmaX)
        self.sigmaY = FloatValue(**self.sigmaY)
        self.brightnessFactor = FloatValue(**self.brightnessFactor)
        self.sigmaFactor = FloatValue(**self.sigmaFactor)



@dataclass
class BiasFrame:
    dataDir: str
    enable: bool

    def __init__(self, **entries):
        self.__dict__.update(entries)

@dataclass
class DarkFrame:
    dataDir: str
    enable: bool

    def __init__(self, **entries):
        self.__dict__.update(entries)

@dataclass
class FlatFrame:
    dataDir: str
    enable: bool

    def __init__(self, **entries):
        self.__dict__.update(entries)


@dataclass
class HotPixel:
    count: int
    brightness: FloatValue
    randomSeed: int
    enable: bool

    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.brightness = FloatValue(**self.brightness)

    def getRandomSeed(self):
        if self.randomSeed == 'random':
            return random.randrange(0, 10000)

        return self.randomSeed

@dataclass
class CosmicRays:
    count: IntValue
    brightness: FloatValue
    pixelCount: IntValue
    spotPixelCount: IntValue
    enable: bool

    cosmicTypes: list

    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.count = IntValue(**self.count)
        self.brightness = FloatValue(**self.brightness)
        self.pixelCount = IntValue(**self.pixelCount)
        self.spotPixelCount = IntValue(**self.spotPixelCount)

        self.cosmicTypes = ['spot', 'track', 'worm']


@dataclass
class Configuration:
    index: int
    realData: DataFile
    numberOfSeries: int
    numberOfFramesInOneSeries: int
    SizeX: int
    SizeY: int
    dataFile: str
    plot: bool
    saveFITSImages: bool
    savePNGImages: bool
    savePositions: bool
    savePixelData: bool
    applyPoisson: bool
    oneFileOnly: bool
    Stars: Stars
    Objects: Objects
    Clusters: Clusters
    Galaxies: Galaxies
    Noise: Noise
    BiasFrame: BiasFrame
    DarkFrame: DarkFrame
    FlatFrame: FlatFrame
    HotPixel: HotPixel
    CosmicRays: CosmicRays

    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.realData = DataFile(**self.realData)
        self.Stars = Stars(**self.Stars)
        self.Objects = Objects(**self.Objects)
        self.Clusters = Clusters(**self.Clusters)
        self.Galaxies = Galaxies(**self.Galaxies)
        self.Noise = Noise(**self.Noise)
        self.BiasFrame = BiasFrame(**self.BiasFrame)
        self.DarkFrame = DarkFrame(**self.DarkFrame)
        self.FlatFrame = FlatFrame(**self.FlatFrame)
        self.HotPixel = HotPixel(**self.HotPixel)
        self.CosmicRays = CosmicRays(**self.CosmicRays)


def loadConfig(filename="config.yml"):
    with open(filename, "r") as ymlfile:
        cfg = yaml.load(ymlfile)
    return Configuration(**cfg)


if __name__ == "__main__":
    with open("config.yml", "r") as ymlfile:
        cfg = yaml.load(ymlfile)

    c = Configuration(**cfg)

    # print(c)
