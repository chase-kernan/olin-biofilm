
from biofilm.classify import train, features
import numpy as np

MASS_WEIGHT = 0.1
CLASS_WEIGHT = 1 - MASS_WEIGHT

class Scorer(object):

    def __init__(self, target_class='anaerobic', target_mass=2500,
                 mass_tolerance=0.1):
        assert target_class in train.CLASSES
        assert target_mass > 0
        assert 0 <= mass_tolerance < 1
        self.target_class = target_class
        self.target_mass = target_mass
        self.mass_tolerance = mass_tolerance

        self.classifier = train.train()
        self._class_index = train.CLASSES[self.target_class]

    def score(self, biofilm_image):
        mass_diff = float(abs(biofilm_image.sum() - self.target_mass))
        if mass_diff/self.target_mass > self.mass_tolerance:
            return MASS_WEIGHT*(1. - mass_diff/self.target_mass)

        score = MASS_WEIGHT
        feat = features.calculate(biofilm_image)
        classes = self.classifier.predict_proba(feat)
        return score + CLASS_WEIGHT*classes[0, self._class_index]

if __name__ == '__main__':
    from biofilm.classify import flat
    image = flat.make_image()
    scorer = Scorer('flat', image.sum())
    print scorer.score(image)

