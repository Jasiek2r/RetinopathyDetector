from abc import abstractmethod


class Metric:
    @abstractmethod
    def perform_measurement(self, y_true, y_pred):
        pass
