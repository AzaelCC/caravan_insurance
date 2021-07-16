from sklearn.ensemble import RandomForestClassifier
from . import utilities as mu

class RandomForestClassifier(RandomForestClassifier):
    def score(self, X, y):
        y_probs = self.predict_proba(X)[:,1]
        hit_rate = mu.model_metrics(y_probs, y)
        return hit_rate