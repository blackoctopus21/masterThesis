from dataclasses import dataclass
import random
import yaml

@dataclass
class Streak:
    length: int
    alpha: int
    sigma: int
    enable: bool

    def __init__(self, **entries):
        self.__dict__.update(entries)

@dataclass
class Value:
    min: int
    max: int
    random: str

    def __init__(self, **entries):
        self.__dict__.update(entries)

    def value(self):
        if self.random == 'uniform':
            return random.randrange(self.min, self.max+1)
        if self.random == 'normal':
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
    count: Value
    brightness: Value
    fwhm: Value
    streak: Streak

    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.count = Value(**self.count)
        self.brightness = Value(**self.brightness)
        self.fwhm = Value(**self.fwhm)
        self.streak = Streak(**self.streak)



@dataclass
class Objects:
    count: Value
    brightness: Value
    fwhm: Value
    speed: Value

    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.count = Value(**self.count)
        self.brightness = Value(**self.brightness)
        self.fwhm = Value(**self.fwhm)
        self.speed = Value(**self.speed)

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
class Configuration:
    numberOfSeries: int
    numberOfFramesInOneSeries: int
    SizeX: int
    SizeY: int
    Stars: Stars
    Objects: Objects
    dataFile: str
    plot: bool
    saveImages: bool
    Noise: Noise
    Bias: Bias
    HotPixel: HotPixel

    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.Stars = Stars(**self.Stars)
        self.Objects = Objects(**self.Objects)
        self.Noise = Noise(**self.Noise)
        self.Bias = Bias(**self.Bias)
        self.HotPixel = HotPixel(**self.HotPixel)


def loadConfig(filename="config.yml"):
    with open(filename, "r") as ymlfile:
        cfg = yaml.load(ymlfile)
    return Configuration(**cfg)


if __name__ == "__main__":
    with open("config.yml", "r") as ymlfile:
        cfg = yaml.load(ymlfile)

    c = Configuration(**cfg)

    # print(c)
