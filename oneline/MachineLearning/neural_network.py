from os.path import join
import torch

from .average_meter import AverageMeter


class NeuralNetwork:

    def __init__(self,
                 model: torch.nn.Module = None,
                 train_dataset: torch.utils.data.DataLoader = None,
                 test_dataset: torch.utils.data.DataLoader = None,
                 optimizer: torch.optim.Optimizer = None,
                 criterion = None,
                 epoch: int = 0,
                 non_optim_epoch: int = None,
                 cpu: bool = False):
        # preparation of base
        self.device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
        self.optimizer = optimizer
        self.model = model.to(self.device) if model else None
        self.criterion = criterion.to(self.device) if criterion else None
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        # the parameters of training
        self.epoch = epoch
        self.train_iter = None
        self.loss = AverageMeter()

        # only for self.auto_train()
        self.non_optim_epoch = non_optim_epoch

    def _train_net(self, info: bool):
        for n in range(self.epoch):
            for i, (source, target) in enumerate(self.train_dataset):
                # send data to device
                source = source.to(self.device)
                target = target.to(self.device)

                # the result and loss
                result = self.model(source)
                loss = self.criterion(result, target)

                # optimization and backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # update the loss
                self.loss.update(loss.item(), source.size(0))

                # print the information
                if info:
                    print(f"\rEpoch: { n } | Batch: { i } | loss: { self.loss.avg }", end="")

            if info:
                print(f"\rEpoch: { n } | Average loss: { self.loss.avg }")

            # reset the loss for a new epoch
            self.loss.reset()

            # for iteration
            yield

    @torch.no_grad()
    def _eval_net(self, info: bool):
        if info:
            print(f"\rEvaluating...", end="")
        eval_total = 0
        eval_correct = 0
        eval_loss = 0
        for i, (source, target) in enumerate(self.test_dataset):
            # send data to device
            source = source.to(self.device)
            target = target.to(self.device)

            result = self.model(source)
            eval_loss += self.criterion(result, target).item()
            _, predicted = torch.max(result.data, 1)
            eval_total += target.size(0)
            eval_correct += (predicted == target).sum().item()

        accuracy = eval_correct / eval_total

        if info:
            print(f"\rEvaluation Accuracy: { accuracy }")

        return eval_loss, accuracy

    def _prepare(self, info: bool):
        self.model.train()
        self.train_iter = self._train_net(info)

        # information display
        if info:
            print(self.model)
            print("=========================")
            print(f"Train data length: { len(self.train_dataset) }")
            print("=========================")

    def auto_train(self,
                   save_model_location: str,
                   eval: bool = True,
                   eval_interval: int = 1,
                   info: bool = True,
                   save_static_dicts: bool = True):
        self._prepare(info)
        best_attempt = float("-inf")
        for n in range(self.epoch):
            next(self.train_iter)
            if self.test_dataset and eval and not n % eval_interval:
                self.model.eval()
                eval_loss, accuracy = self._eval_net(info)
                if accuracy > best_attempt:
                    best_attempt = accuracy
                    if save_static_dicts:
                        self.save_state_dict(join(save_model_location, 'model.pkl'))
                    else:
                        self.save_model(join(save_model_location, 'model.pkl'))

    def iter_train(self, info: bool = True):
        if not self.train_iter:
            self._prepare(info)
        return next(self.train_iter)

    def save_state_dict(self, location: str):
        torch.save(self.model.state_dict(), location)

    def save_model(self, location: str):
        torch.save(self.model, location)
