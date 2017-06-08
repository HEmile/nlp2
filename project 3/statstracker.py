from matplotlib import pyplot as plt

class StatsTracker:
    def __init__(self):
        self.n = []
        self.loss = []
        self.acc = []
        self.aer = []
        self.plot()

    def update(self,n, loss, acc, aer):
        self.n.append(n)
        self.loss.append(loss)
        self.acc.append(acc)
        self.aer.append(aer)
        self.update_plot()

    def plot(self):
        plt.figure(1)
        ax = plt.gca()
        if ax.lines:
            ax.lines.remove(ax.lines[0])
        plt.title("Loss over epochs ")
        plt.plot(self.n, self.loss)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.figure(2)
        ax = plt.gca()
        if ax.lines:
            ax.lines.remove(ax.lines[0])
        plt.title("Accuracy over epochs ")
        plt.plot(self.n, self.acc)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')

        plt.figure(3)
        ax = plt.gca()
        if ax.lines:
            ax.lines.remove(ax.lines[0])
        plt.title("AER over epochs")
        plt.plot(self.n, self.aer)
        plt.xlabel('Epochs')
        plt.ylabel('AER')

    def update_plot(self):
        plt.figure(1)
        ax = plt.gca()
        if ax.lines:
            ax.lines.remove(ax.lines[0])
        plt.plot(self.n, self.loss)

        plt.figure(2)
        ax = plt.gca()
        if ax.lines:
            ax.lines.remove(ax.lines[0])
        plt.plot(self.n, self.acc)

        plt.figure(3)
        ax = plt.gca()
        if ax.lines:
            ax.lines.remove(ax.lines[0])
        plt.plot(self.n, self.aer)