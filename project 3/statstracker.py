from matplotlib import pyplot as plt

class StatsTracker:
    def __init__(self):
        self.n = []
        self.loss = []
        self.acc = []
        self.lr = []
        self.epoch_nr = 0
        self.plot()

    def update(self,n, loss, acc, lr):
        self.n.append(n)
        self.loss.append(loss)
        self.acc.append(acc)
        self.lr.append(lr)
        self.update_plot()

    def plot(self):
        plt.figure(self.epoch_nr * 3 + 1)
        plt.gcf().clear()
        plt.title("Loss over batches in epoch " + str(self.epoch_nr + 1))
        plt.plot(self.n, self.loss)
        plt.xlabel('Batch number')
        plt.ylabel('Loss')

        plt.figure(self.epoch_nr * 3 + 2)
        plt.gcf().clear()
        plt.title("Accuracy over batches in epoch " + str(self.epoch_nr + 1))
        plt.plot(self.n, self.acc)
        plt.xlabel('Batch number')
        plt.ylabel('Accuracy')

        plt.figure(self.epoch_nr * 3 + 3)
        plt.gcf().clear()
        plt.title("Learning rate over batches in epoch " + str(self.epoch_nr + 1))
        plt.plot(self.n, self.lr)
        plt.xlabel('Batch number')
        plt.ylabel('Learning rate')

    def update_plot(self):
        plt.figure(self.epoch_nr * 3 + 1)
        plt.gcf().clear()
        plt.plot(self.n, self.loss)

        plt.figure(self.epoch_nr * 3 + 2)
        plt.gcf().clear()
        plt.plot(self.n, self.acc)

        plt.figure(self.epoch_nr * 3 + 3)
        plt.gcf().clear()
        plt.plot(self.n, self.lr)

    def update_epoch(self, epoch):
        self.epoch_nr = epoch - 1
        self.n = []
        self.loss = []
        self.acc = []
        self.lr = []
        self.plot()