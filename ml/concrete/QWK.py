from sklearn.metrics import cohen_kappa_score

from ml.abstractions.metric import Metric


class QWK(Metric):
    def perform_measurement(self, y_true, y_pred):
        return cohen_kappa_score(y_true, y_pred, weights="quadratic")