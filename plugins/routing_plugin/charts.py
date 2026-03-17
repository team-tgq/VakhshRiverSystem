# plugins/routing_plugin/charts.py
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class LossChart(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 4), dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)

    def plot_losses(self, train_losses, val_losses=None, title="Loss Curve"):
        self.ax.clear()

        if train_losses:
            self.ax.plot(train_losses, label="Train Loss")
        if val_losses:
            self.ax.plot(val_losses, label="Val Loss")

        self.ax.set_title(title)
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.draw()


class PredictionChart(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(7, 4), dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)

    def plot_prediction(
        self,
        y_true,
        y_pred,
        title="Prediction vs Actual",
        true_label="Actual",
        pred_label="Pred"
    ):
        self.ax.clear()

        self.ax.plot(y_true, label=true_label)
        self.ax.plot(y_pred, label=pred_label)

        self.ax.set_title(title)
        self.ax.set_xlabel("Time Step")
        self.ax.set_ylabel("Value")
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.draw()