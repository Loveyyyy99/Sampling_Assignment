from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

def sampling1(X, y):
    return RandomUnderSampler(random_state=42).fit_resample(X, y)

def sampling2(X, y):
    return RandomOverSampler(random_state=42).fit_resample(X, y)

def sampling3(X, y):
    return SMOTE(random_state=42).fit_resample(X, y)

def sampling4(X, y):
    return SMOTEENN(random_state=42).fit_resample(X, y)

def sampling5(X, y):
    return SMOTETomek(random_state=42).fit_resample(X, y)
