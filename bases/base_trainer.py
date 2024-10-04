import abc

class BaseTrainer(abc.ABC):
    @property
    @abc.abstractmethod
    def total_epochs(self):
        pass

    @property
    @abc.abstractmethod
    def epoch_start(self):
        pass

    @abc.abstractmethod
    def get_model(self):
        pass

    @abc.abstractmethod
    def get_dataloader(self):
        pass

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def eval(self):
        pass

    @abc.abstractmethod
    def save(self, epoch):
        pass

    @abc.abstractmethod
    def load(self, epoch):
        pass

    @abc.abstractmethod
    def tensorboard_log(self, epoch, train_loss, val_loss):
        pass

    def run(self):
        self.get_model()
        self.get_dataloader()

        for epoch in range(self.epoch_start, self.total_epochs+1):
            train_loss = self.train()
            test_loss = self.eval()

            self.tensorboard_log(epoch, train_loss, test_loss)
