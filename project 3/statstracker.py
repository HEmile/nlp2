from matplotlib import pyplot as plt

class StatsTracker:
    def __init__(self):
        self.n = []
        self.loss = []
        self.acc = []
        self.aer = []
        self.epoch_nr = 0
        self.plot()

    def update(self,n, loss, acc, aer):
        self.n.append(n)
        self.loss.append(loss)
        self.acc.append(acc)
        self.aer.append(aer)
        self.update_plot()

    def plot(self):
        plt.figure(self.epoch_nr * 3 + 1)
        ax = plt.gca()
        if ax.lines:
            ax.lines.remove(ax.lines[0])
        plt.title("Loss over epochs ")
        plt.plot(self.n, self.loss)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.figure(self.epoch_nr * 3 + 2)
        ax = plt.gca()
        if ax.lines:
            ax.lines.remove(ax.lines[0])
        plt.title("Accuracy over epochs ")
        plt.plot(self.n, self.acc)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')

        plt.figure(self.epoch_nr * 3 + 3)
        ax = plt.gca()
        if ax.lines:
            ax.lines.remove(ax.lines[0])
        plt.title("AER over epochs")
        plt.plot(self.n, self.aer)
        plt.xlabel('Epochs')
        plt.ylabel('AER')

    def update_plot(self):
        plt.figure(self.epoch_nr * 3 + 1)
        ax = plt.gca()
        ax.lines.remove(ax.lines[0])
        plt.plot(self.n, self.loss)

        plt.figure(self.epoch_nr * 3 + 2)
        ax = plt.gca()
        ax.lines.remove(ax.lines[0])
        plt.plot(self.n, self.acc)

        plt.figure(self.epoch_nr * 3 + 3)
        ax = plt.gca()
        ax.lines.remove(ax.lines[0])
        plt.plot(self.n, self.aer)

    def update_epoch(self, epoch):
        self.epoch_nr = epoch - 1
        self.n = []
        self.loss = []
        self.acc = []
        self.lr = []
        self.plot()