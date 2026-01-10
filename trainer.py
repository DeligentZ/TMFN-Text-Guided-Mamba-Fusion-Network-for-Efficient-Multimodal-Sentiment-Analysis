import os
import torch
import time
from torch import nn
import pandas as pd
from tqdm import tqdm
from utils.metricsTop import MetricsTop
from torch.cuda.amp import autocast, GradScaler

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class Trainer:
    def __init__(self, config):
        self.config = config
        self.dataset_name = config["dataset_name"]
        self.model_save_path = config["model_save_path"]

        self.mse = torch.nn.functional.mse_loss
        self.metrics = MetricsTop("regression").getMetics(config["dataset_name"])
        self.scaler = GradScaler()
        self.best_epoch = 0
        self.lowest_eval_loss = float('inf')
        self.highest_eval_acc = 0
        self.num_model = 0
        self.loss_weight = 0
        self.current_step = 0
        self.warmup_steps = 0
        self.epoch_steps = 0
        self.initial_lr = 0
        self.epoch = 0

        if self.dataset_name == 'ur_funny':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.L1Loss()
    def train_epoch(self, model, loader, optimizer):
        model.train()
        total_loss = 0

        for batch in tqdm(loader, desc = "Training"):
            optimizer.zero_grad()
            inputs = self._inputs(batch)
            targets = batch["targets"].to(device).view(-1, 1)
            with (autocast()):
                outputs = model(*inputs)
                loss = self.criterion(outputs, targets)
                loss_mse = self.mse(outputs, targets)
                loss = loss + 0.03 * loss_mse
            self.scaler.scale(loss).backward() # scale the loss and compute the gradient.
            self.scaler.unscale_(optimizer) # unscale the loss for clop_gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            self.current_step += 1
            if self.warmup_steps > 0 and self.current_step <= self.warmup_steps:
                warmup_scale = float(self.current_step) / float(self.warmup_steps)
                for g in optimizer.param_groups:
                    g["lr"] = self.initial_lr * warmup_scale
            self.scaler.step(optimizer)
            self.scaler.update()

            total_loss += loss.item() * targets.size(0)
        avg_loss = total_loss / len(loader.dataset)
        print('%s: >>' % f"total loss : {avg_loss:.4f}")
        return round(avg_loss, 4)

    def evaluate(self, model, loader, mode = "VAl"):
        model.eval()
        y_pred, y_true, total_loss = [], [], 0
        with torch.no_grad():
            for batch in tqdm(loader, desc = mode):
                inputs = self._inputs(batch)
                targets = batch["targets"].to(device).view(-1, 1)
                outputs = model(*inputs)
                loss = self.criterion(outputs, targets)
                loss_mse = self.mse(outputs, targets)
                loss = loss + 0.03 * loss_mse
                total_loss += loss.item() * targets.size(0)
                y_pred.append(outputs.cpu())
                y_true.append(targets.cpu())
        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)
        eval_results = self.metrics(y_pred, y_true)
        eval_results['Loss'] = round(total_loss / len(loader.dataset), 4)
        print('%s: >> ' % f"{mode}" + _dict_to_str(eval_results))
        #print('%s: >>' % f"total loss : {loss}, loss_mae: {loss_mae}  loss_cls : {0.3 * loss_cls} ")
        print('%s: >>' % f"total loss : {round(total_loss / len(loader.dataset), 4)}")
        return eval_results

    def fit(self, model, train_loader, val_loader, test_loader):

        optimizer = torch.optim.AdamW(model.parameters(), lr=float(self.config["lr"]), weight_decay=0.01)
        self.initial_lr = float(self.config["lr"])
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * self.config["epochs"]
        if self.dataset_name == "mosi":
            warmup_ratio = 0.03  # 3%
        elif self.dataset_name == "sims":
            warmup_ratio = 0.05  # 5%
        elif self.dataset_name == "mosei":
            warmup_ratio = 0.06  # 6%
        else:
            warmup_ratio = 0.1
        self.warmup_steps = int(total_steps * warmup_ratio)
        train_loss_log, val_loss_log, val_acc_log = [], [], []
        while True:
            print(f"----------------- EPOCH {self.epoch} -----------------")
            self.epoch += 1
            train_loss = self.train_epoch(model, train_loader, optimizer)
            eval_results = self.evaluate(model, val_loader, mode = "VAL")
            train_loss_log.append(train_loss)
            val_loss_log.append(eval_results['Loss'])
            if self.dataset_name == "sims":
                val_acc_log.append(eval_results['Mult_acc_2'])
            elif self.dataset_name == "ur_funny":
                val_acc_log.append(eval_results["Acc"])
            else:
                val_acc_log.append(eval_results["Has0_acc_2"])
            self._save_best(model, eval_results, self.epoch, self.dataset_name)
            if self.epoch - self.best_epoch >= self.config["early_stop"]:
                break

        return self._evaluate_and_plot(model, test_loader, self.dataset_name, train_loss_log, val_loss_log, val_acc_log)


    def _inputs(self, batch):
        return (
            batch["text_tokens"].to(device),
            batch["text_masks"].to(device),
            batch["audio_inputs"].to(device),
            batch["audio_masks"].to(device),
            batch["video_features"].to(device),
        )
    def _save_best(self, model, eval_results, epoch, dataset_name):
        if dataset_name == "sims":
            acc2, acc5 = 0.74, 0.40
            eval_loss, eval_acc, eval_acc5 = eval_results["Loss"], eval_results["Mult_acc_2"], eval_results["Mult_acc_5"]
            if eval_loss < self.lowest_eval_loss:
                self.lowest_eval_loss = eval_loss
                torch.save(model.state_dict(), os.path.join(self.model_save_path, f"{dataset_name}_best_loss.pth"))
                self.best_epoch = epoch
            if eval_acc >= self.highest_eval_acc:
                self.highest_eval_acc = eval_acc
                torch.save(model.state_dict(), os.path.join(self.model_save_path, f"{dataset_name}_best_acc.pth"))
            if eval_acc >= acc2 and eval_acc5 >= acc5:
                torch.save(model.state_dict(), os.path.join(self.model_save_path, f"{dataset_name}_best_acc2_5_{self.num_model}.pth"))
                self.num_model += 1
        elif dataset_name == "ur_funny":
            eval_loss, eval_acc= eval_results['Loss'], eval_results["Acc"]
            if eval_loss < self.lowest_eval_loss:
                self.lowest_eval_loss = eval_loss
                torch.save(model.state_dict(), os.path.join(self.model_save_path, f"{dataset_name}_best_loss.pth"))
                self.best_epoch = epoch
            if eval_acc >= self.highest_eval_acc:
                self.highest_eval_acc = eval_acc
                torch.save(model.state_dict(), os.path.join(self.model_save_path, f"{dataset_name}_best_acc.pth"))
        else:
            eval_loss, eval_acc, eval_acc7 = eval_results['Loss'], eval_results["Has0_acc_2"], eval_results["Mult_acc_7"]
            if eval_loss < self.lowest_eval_loss:
                self.lowest_eval_loss = eval_loss
                torch.save(model.state_dict(), os.path.join(self.model_save_path, f"{dataset_name}_best_loss.pth"))
                self.best_epoch = epoch
            if eval_acc >= self.highest_eval_acc:
                self.highest_eval_acc = eval_acc
                torch.save(model.state_dict(), os.path.join(self.model_save_path, f"{dataset_name}_best_acc.pth"))
            if self.dataset_name == 'mosi':
                acc2, acc7 = 0.80, 0.45
            else: # cmu-mosei
                acc2, acc7 = 0.82, 0.53
            if eval_acc >= acc2 and eval_acc7 >= acc7:
                torch.save(model.state_dict(), os.path.join(self.model_save_path, f"{dataset_name}_best_acc2_7_{self.num_model}.pth"))
                self.num_model += 1
    def _evaluate_and_plot(self, model, test_loader, dataset_name,train_loss_log, val_loss_log, val_acc_log):
        result_all = {}
        if dataset_name == "sims":
            # load the lowest loss model for testing
            model.load_state_dict(torch.load(os.path.join(self.model_save_path, f"{dataset_name}_best_loss.pth")))
            test_results_loss = self.evaluate(model, test_loader, mode = "Test (lowest loss)")
            result_all["lowest_loss"] = test_results_loss
            print('%s: >> ' % ('Test (lowest loss)') + _dict_to_str(test_results_loss))
            # load the highest acc model for testing
            model.load_state_dict(torch.load(os.path.join(self.model_save_path, f"{dataset_name}_best_acc.pth")))
            test_results_acc = self.evaluate(model, test_loader, mode = "Test (highest acc)")
            print('%s: >> ' % ('TEST (highest acc)') + _dict_to_str(test_results_acc))
            result_all["highest acc"] = test_results_acc
            for index in range(self.num_model):
                model.load_state_dict(torch.load(os.path.join(self.model_save_path, f"{dataset_name}_best_acc2_5_{index}.pth")))
                test_results_loss_mix = self.evaluate(model, test_loader, mode = "Test (Mix)")
                print('\n%s: >> ' % (f'TEST (best mix)[{index}] ') + _dict_to_str(test_results_loss_mix))
                result_all[f"mix_{index}"] = test_results_loss_mix
        else:
            # load the lowest loss model for testing
            model.load_state_dict(torch.load(os.path.join(self.model_save_path, f"{dataset_name}_best_loss.pth")))
            test_results_loss = self.evaluate(model, test_loader, mode="Test (lowest loss)")
            result_all["lowest_loss"] = test_results_loss
            #error_analyzer = self._error_analyzer(model, test_loader, save_path="test_prediction_lowest_Loss.csv")
            print('%s: >> ' % ('Test (lowest loss)') + _dict_to_str(test_results_loss))
            # load the highest acc model for testing
            model.load_state_dict(torch.load(os.path.join(self.model_save_path, f"{dataset_name}_best_acc.pth")))
            test_results_acc = self.evaluate(model, test_loader, mode="Test (highest acc)")
            #error_analyzer = self._error_analyzer(model, test_loader, save_path="test_prediction_highest acc.csv")
            print('%s: >> ' % ('TEST (highest acc)') + _dict_to_str(test_results_acc))
            result_all["highest acc"] = test_results_acc
            for index in range(self.num_model):
                model.load_state_dict(torch.load(os.path.join(self.model_save_path, f"{dataset_name}_best_acc2_7_{index}.pth")))
                test_results_loss_mix = self.evaluate(model, test_loader, mode="Test (Mix)")
                print('\n%s: >> ' % (f'TEST (best mix)[{index}] ') + _dict_to_str(test_results_loss_mix))
                result_all[f"mix_{index}"] = test_results_loss_mix
        return result_all
    def _error_analyzer(self, model, loader, save_path="val_predictions_detail.csv"):
        model.eval()
        all_records = []
        with torch.no_grad():
            for batch in tqdm(loader, desc = "Logging predictions"):
                inputs = self._inputs(batch)
                targets = batch["targets"].to(device).view(-1, 1)
                ids = batch["ids"]
                #outputs, output_cls = model(*inputs)
                outputs = model(*inputs)
                pred_mae = outputs.squeeze()
                for i in range(len(pred_mae)):
                    all_records.append({
                        "id": ids[i],
                        "pred_mae": pred_mae[i].item(),
                        "target": targets[i].item(),
                        "error_mae":abs(pred_mae[i].item() - targets[i].item()),
                    })
        df = pd.DataFrame(all_records)
        df.to_csv(save_path, index=False)
        print(f"✅ Saved prediction details to {save_path}")

    def _error_analyzer(self, model, loader, save_path="val_predictions_detail.csv"):
        model.eval()
        all_records = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Logging predictions"):
                inputs = self._inputs(batch)
                targets = batch["targets"].to(device).view(-1, 1)
                ids = batch["ids"]
                outputs = model(*inputs)
                pred_mae = outputs.squeeze()
                for i in range(len(pred_mae)):
                    all_records.append({
                        "id": ids[i],
                        "pred_mae": pred_mae[i].item(),
                        "target": targets[i].item(),
                        "error_mae": abs(pred_mae[i].item() - targets[i].item()),
                    })
        df = pd.DataFrame(all_records)
        df.to_csv(save_path, index=False)
        print(f"✅ Saved prediction details to {save_path}")
def _dict_to_str( d):
    return ' '.join([f'{k}: {v:.4f}' for k, v in d.items()])


