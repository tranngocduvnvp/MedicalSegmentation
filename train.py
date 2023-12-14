import torch
import time
import os
import numpy as np
from eval import validation, test
from build import build
import argparse
import yaml
import sys

def train_epoch(model, device, train_loader, optimizer, epoch, Dice_loss, BCE_loss, args):
    t = time.time()
    model.train()
    loss_accumulator = []
    for batch_idx, (data, target, _) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        for k in range(0, data.shape[0], args.mini_batch):
            data_input = data[k:k + args.mini_batch]
            target_input = target[k:k + args.mini_batch]
            output = model(data_input)
            loss = Dice_loss(output, target_input) + BCE_loss(torch.sigmoid(output), target_input)
            loss.backward()
        optimizer.step()
        loss_accumulator.append(loss.item())
        if batch_idx + 1 < len(train_loader):
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tTime: {:.6f}".format(
                    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset), 100.0 * (batch_idx + 1) / len(train_loader),
                    loss.item(), time.time() - t, ), end="", )
        else:
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tAverage loss: {:.6f}\tTime: {:.6f}".format(
                    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset), 100.0 * (batch_idx + 1) / len(train_loader),
                    np.mean(loss_accumulator), time.time() - t, ) )

    return np.mean(loss_accumulator)



def train(args):

    if not os.path.exists("./Trained models"):
        os.makedirs("./Trained models")

    ( device, train_dataloader, val_dataloader, test_dataloader,
     perf, model, optimizer, checkpoint, scheduler, loss_fun) = build(args)

    prev_best_test = checkpoint["test_measure_mean"]
    print("best test:", prev_best_test, "epoch:", checkpoint["epoch"])

    for epoch in range(1, args.epochs + 1):
        try:
            loss = train_epoch(
                model, device, train_dataloader, optimizer, epoch, loss_fun["Dice_loss"],\
                      loss_fun["BCE_loss"], args
            )
            val_measure_mean, val_measure_std = validation(
                model, device, val_dataloader, epoch, perf,"Val"
            )
            test_measure_mean, test_measure_std = test(
                model, device, test_dataloader, epoch, perf,"Test"
            )
        except KeyboardInterrupt:
            print("Training interrupted by user")
            sys.exit(0)
        if args.lrs == "true":
            if args.type_lr == "LROnP":
                scheduler.step(test_measure_mean)
            else:
                scheduler.step()
        if prev_best_test == None or val_measure_mean > prev_best_test:
            print('\033[41m' + "Saving..." + '\033[0m')
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict()
                    if args.mgpu == "false"
                    else model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler":scheduler.state_dict(),
                    "loss": loss,
                    "test_measure_mean": test_measure_mean,
                    "test_measure_std": test_measure_std,
                },
                f"./Trained models/" + args.dataset + "_" + args.model_name + ".pt",
            )
            prev_best_test = val_measure_mean


def main(args):
    train(args)


if __name__ == "__main__":
    # Đường dẫn đến tệp YAML
    yaml_file = "/home/bigdata/Documents/TND_Modeling/config.yaml"

    # Đọc tệp YAML
    with open(yaml_file, "r") as file:
        yaml_data = yaml.safe_load(file)

    # Chuyển đổi dữ liệu YAML thành đối tượng namespace
    args = argparse.Namespace(**yaml_data)
    main(args)