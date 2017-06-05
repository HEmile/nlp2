from matplotlib import pyplot as plt

class StatsTracker:
    def __init__(self):
        self.n = []
        self.loss = []
        self.acc = []
        self.lr = []

    def update(self,n, loss, acc, lr):
        self.n.append(n)
        self.loss.append(loss)
        self.acc.append(acc)
        self.lr.append(lr)

    def plot(self):
        plt.figure(1)
        plt.title("Loss over batches")
        plt.plot(self.n, self.loss)
        plt.xlabel('Batch number')
        plt.ylabel('Loss')


        plt.figure(2)
        plt.title("Accuracy over batches")
        plt.plot(self.n, self.acc)
        plt.xlabel('Batch number')
        plt.ylabel('Accuracy')

        plt.figure(3)
        plt.title("Learning rate over batches")
        plt.plot(self.n, self.lr)
        plt.xlabel('Batch number')
        plt.ylabel('Learning rate')