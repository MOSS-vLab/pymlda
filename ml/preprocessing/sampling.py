from imblearn.under_sampling import RandomUnderSampler

class UnderSampler:
    
    def __init__(self):
        self.rus = RandomUnderSampler()
    
    def fit_resample(self, X, y):
        return self.rus.fit_resample(X, y)