from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

def sampling1(X, y):
    """Random Under Sampling"""
    sampler = RandomUnderSampler(random_state=42)
    return sampler.fit_resample(X, y)

def sampling2(X, y):
    """Random Over Sampling"""
    sampler = RandomOverSampler(random_state=42)
    return sampler.fit_resample(X, y)

def sampling3(X, y):
    """SMOTE"""
    sampler = SMOTE(random_state=42)
    return sampler.fit_resample(X, y)

def sampling4(X, y):
    """SMOTE + ENN"""
    sampler = SMOTEENN(random_state=42)
    return sampler.fit_resample(X, y)

def sampling5(X, y):
    """SMOTE + Tomek"""
    sampler = SMOTETomek(random_state=42)
    return sampler.fit_resample(X, y)
