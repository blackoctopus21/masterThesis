from dataclasses import dataclass
import random
import yaml

@dataclass
class DataFile:
    enabled: bool
    file: str

    def __init__(self, **entries):
        self.__dict__.update(entries)


@dataclass
class Value:
    min: int
    max: int
    random: str
    savedValue: int = None

    def __init__(self, **entries):
        self.__dict__.update(entries)

    def uniform(self):
        return random.randrange(self.min, self.max + 1)

    def normal(self):
        return int(random.gauss(mu=(abs(self.min + self.max) / 2), sigma=(self.max - self.min)))

    def value(self):
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


@dataclass
class Noise:
    mean: int
    std: int
    enable: bool

    def __init__(self, **entries):
        self.__dict__.update(entries)


@dataclass
class Stars:
    count: Value
    brightness: Value
    fwhm: Value
    method: str
    length: Value
    alpha: Value
    realData: DataFile

    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.count = Value(**self.count)
        self.brightness = Value(**self.brightness)
        self.fwhm = Value(**self.fwhm)
        self.length = Value(**self.length)
        self.alpha = Value(**self.alpha)
        self.realData = DataFile(**self.realData)


@dataclass
class Objects:
    count: Value
    brightness: Value
    fwhm: Value
    speed: Value
    method: str
    length: Value
    alpha: Value
    enable: bool

    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.count = Value(**self.count)
        self.brightness = Value(**self.brightness)
        self.fwhm = Value(**self.fwhm)
        self.speed = Value(**self.speed)
        self.length = Value(**self.length)
        self.alpha = Value(**self.alpha)


@dataclass
class Clusters:
    count: Value
    objectCountPerCluster: Value
    brightness: Value
    fwhm: Value
    speed: Value
    method: str
    length: Value
    alpha: Value
    enable: bool

    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.count = Value(**self.count)
        self.objectCountPerCluster = Value(**self.objectCountPerCluster)
        self.brightness = Value(**self.brightness)
        self.fwhm = Value(**self.fwhm)
        self.speed = Value(**self.speed)
        self.length = Value(**self.length)
        self.alpha = Value(**self.alpha)


@dataclass
class Bias:
    value: int
    columns: int
    enable: bool

    def __init__(self, **entries):
        self.__dict__.update(entries)


@dataclass
class HotPixel:
    count: int
    brightness: Value
    enable: bool

    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.brightness = Value(**self.brightness)


@dataclass
class CosmicRays:
    count: Value
    brightness: Value
    pixelCount: Value
    spotPixelCount: Value
    enable: bool

    cosmicTypes: list

    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.count = Value(**self.count)
        self.brightness = Value(**self.brightness)
        self.pixelCount = Value(**self.pixelCount)
        self.spotPixelCount = Value(**self.spotPixelCount)

        self.cosmicTypes = ['spot', 'track', 'worm']


@dataclass
class Configuration:
    realData: DataFile
    numberOfSeries: int
    numberOfFramesInOneSeries: int
    SizeX: int
    SizeY: int
    dataFile: str
    plot: bool
    saveImages: bool
    Stars: Stars
    Objects: Objects
    Clusters: Clusters
    Noise: Noise
    Bias: Bias
    HotPixel: HotPixel
    CosmicRays: CosmicRays

    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.realData = DataFile(**self.realData)
        self.Stars = Stars(**self.Stars)
        self.Objects = Objects(**self.Objects)
        self.Clusters = Clusters(**self.Clusters)
        self.Noise = Noise(**self.Noise)
        self.Bias = Bias(**self.Bias)
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
