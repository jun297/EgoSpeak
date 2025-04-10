import argparse
import os
import os.path as osp

import torch
import wandb
import yaml
from criterions import build_criterion
from model import build_model
from torch.utils.tensorboard import SummaryWriter
from trainer import build_eval, build_trainer
from utils import *
from utils import get_logger

from datasets import build_data_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--lr_scheduler", action="store_true")
    parser.add_argument("--no_rgb", action="store_true")
    parser.add_argument("--no_flow", action="store_true")
    parser.add_argument(
        "--save-as-cfg",
        action="store_true",
        help="Save outputs with config name in the output directory",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--save-preds",
        action="store_true",
        help="Save predictions in the output directory",
    )
    # parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()

    # combine argparse and yaml
    opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
    opt.update(vars(args))
    cfg = opt

    if "checkpoint" in cfg and cfg["checkpoint"] is not None and args.ckpt is None:
        args.ckpt = cfg["checkpoint"]
        print(f"checkpoint is given in the config, set args.ckpt to {args.ckpt}")

    seed = 20 if "seed" not in cfg else cfg["seed"]
    set_seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    annotation_type = os.path.basename(cfg["annotation_type"])

    identifier = os.path.basename(cfg["config"])[:-5]
    identifier += "_"
    if cfg["no_flow"] == False:
        identifier += "A"
    if cfg["no_rgb"] == False:
        identifier += "V"

    if cfg["save_as_cfg"] == True:
        output_dir = osp.join("output_cfg", identifier)
        result_path = create_outdir(output_dir, overwrite=True)
    else:
        identifier = get_identifier(cfg)
        annotation_type = os.path.basename(cfg["annotation_type"])
        identifier += f"_{annotation_type}"
        result_path = create_outdir(osp.join(cfg["output_path"], identifier))

    logger = get_logger(result_path)
    logger.info(f"Identifier: {identifier}")
    logger.info(f"result_path: {result_path}")
    logger.info(cfg)

    testloader = build_data_loader(cfg, mode="test")
    print("testloader loaded")
    model = build_model(cfg, device)
    evaluate = build_eval(cfg)

    trainloader = build_data_loader(cfg, mode="train")
    print("trainloader loaded")
    
    # evaluation
    if args.eval == True and cfg["ckpt"] == None:
        print("Automatically find the best checkpoint")
        ckpt_root = os.path.join("output_cfg", identifier, "ckpts")
        ckpt_dirs = os.listdir(ckpt_root)

        ckpt_path = os.path.join(ckpt_root, "best.pth")
        if os.path.exists(ckpt_path):
            args.ckpt = ckpt_path
            print(f"args.ckpt set to {args.ckpt}")

        else:
            ckpt_path = os.path.join(ckpt_root, "pred", "best.pth")
            if os.path.exists(ckpt_path):
                args.ckpt = ckpt_path
                print(f"args.ckpt set to {args.ckpt}")

        print("Warninig) no ckpt_path found, there can be multiple ckpts")
        print(f"ckpt_dirs: {ckpt_dirs}")
        pred_dir_number = 0
        ckpt_dirs.sort(reverse=True)

        if args.ckpt == None:
            for ckpt_dir in ckpt_dirs:
                ckpt_path = os.path.join(ckpt_root, ckpt_dir, "best.pth")

                if os.path.exists(ckpt_path):
                    args.ckpt = ckpt_path
                    print(f"Found ckpt, args.ckpt was None, set to {args.ckpt}")
                    break

        if args.ckpt == None:
            raise ValueError("No checkpoint found, args.ckpt is None")

    # load checkpoint if args.ckpt is given
    if args.eval == True:
        if args.ckpt == None:
            raise ValueError("Please specify the checkpoint path")
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint)

    elif args.eval == False and args.ckpt != None:
        checkpoint = torch.load(args.ckpt)
        print(f"Load checkpoint from {args.ckpt}")
        model_dict = model.state_dict()

        unnecessary_keys = []
        matched_ckpt = {}
        # Filter out unnecessary keys
        for k, v in checkpoint.items():
            if not (k in model_dict and model_dict[k].shape == checkpoint[k].shape):
                unnecessary_keys.append(k)
            elif (
                "except_classifier" in cfg
                and cfg["except_classifier"] == True
                and "classification" in k
            ):
                print(f"Unnecessary key: {k} is classifier")
                unnecessary_keys.append(k)
            else:
                matched_ckpt[k] = v

        # Overwrite entries in the existing state dict
        model_dict.update(matched_ckpt)

        # Load the new state dict
        model.load_state_dict(model_dict)

        print(f"Unnecessary keys: {unnecessary_keys}")
        
    if cfg["wandb"] == True:
        # args.tensorboard = True
        print("wandb init")
        os.environ["WANDB__SERVICE_WAIT"] = "3000"
        wandb.init(
            project="egospeak",
            tags=[cfg["model"], cfg["data_name"], annotation_type],
            config=cfg,
            name=os.path.basename(result_path),
        )

    if args.eval == True or args.debug == True:
        if args.eval == True and args.ckpt == None:
            raise ValueError("Please specify the checkpoint path")

        logger.info(f'Dataset: {cfg["data_name"]},  Model: {cfg["model"]}')
        logger.info(f"Result path:{result_path}")

        for phase in ["test"]:
            if phase == "train":
                raise NotImplementedError("evaluation on train split is not implemented")
                dataloader = trainloader
                
            elif phase == "test":
                dataloader = testloader

            mAP, mcAP, preds = evaluate(
                model, dataloader, logger, result_path=result_path
            )

            pred_path = osp.join(result_path, "eval")
            os.makedirs(pred_path, exist_ok=True)

            if args.save_preds == True:
                print(f"pred_path: {pred_path}")
                for k, v in preds.items():
                    np.save(osp.join(pred_path, f"{k}.npy"), v)

            logger.info(f'{phase}) {cfg["task"]} result: {mAP*100:.2f} mAP')
            logger.info(f'{phase}) {cfg["task"]} result: {mcAP*100:.2f} mcAP')
        exit()

    criterion = build_criterion(cfg, device)
    train_one_epoch = build_trainer(cfg)
    optim = torch.optim.AdamW if cfg["optimizer"] == "AdamW" else torch.optim.Adam
    optimizer = optim(
        [{"params": model.parameters(), "initial_lr": cfg["lr"]}],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    scheduler = (
        build_lr_scheduler(cfg, optimizer, len(trainloader))
        if args.lr_scheduler
        else None
    )
    writer = SummaryWriter(osp.join(result_path, "runs")) if args.tensorboard else None
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    total_params = sum(p.numel() for p in model.parameters())

    logger.info(f'Dataset: {cfg["data_name"]},  Model: {cfg["model"]}')
    logger.info(
        f'lr:{cfg["lr"]} | Weight Decay:{cfg["weight_decay"]} | Window Size:{cfg["window_size"]} | Batch Size:{cfg["batch_size"]}'
    )
    logger.info(
        f'Total epoch:{cfg["num_epoch"]} | Total Params:{total_params/1e6:.1f} M | Optimizer: {cfg["optimizer"]}'
    )
    logger.info(f"Output Path:{result_path}")

    best_mAP, best_epoch = 0, 0
    best_cAP, best_cAP_epoch = 0, 0

    pred_save_path = osp.join(result_path, "ckpts", "pred")
    print(f"pred_save_path: {pred_save_path}")

    # if you want to overwrite, change the code
    os.makedirs(pred_save_path, exist_ok=True)

    for epoch in range(1, cfg["num_epoch"] + 1):
        epoch_loss = train_one_epoch(
            trainloader,
            model,
            criterion,
            optimizer,
            scaler,
            epoch,
            cfg["wandb"],
            writer,
            scheduler=scheduler,
        )
        trainloader.dataset._init_features()

        mAP, cAP, preds = evaluate(
            model, testloader, logger, criterion, cfg["wandb"], epoch, writer
        )
        
        if mAP > best_mAP:
            best_mAP = mAP
            best_epoch = epoch
            torch.save(model.state_dict(), osp.join(pred_save_path, "best.pth"))
            
            if args.save_preds == True:
                for k, v in preds.items():
                    np.save(osp.join(pred_save_path, f"{k}.npy"), v)

        if cAP > best_cAP:
            best_cAP = cAP
            best_cAP_epoch = epoch
            torch.save(model.state_dict(), osp.join(pred_save_path, "best_cAP.pth"))

        logger.info(
            f'Epoch {epoch} mAP: {mAP*100:.2f} | Best mAP: {best_mAP*100:.2f} at epoch {best_epoch}, iter {epoch*cfg["batch_size"]*len(trainloader)} | train_loss: {epoch_loss/len(trainloader):.4f}, lr: {optimizer.param_groups[0]["lr"]:.7f}'
        )
        logger.info(
            f'Epoch {epoch} cAP: {cAP*100:.2f} | Best cAP: {best_cAP*100:.2f} at epoch {best_epoch}, iter {epoch*cfg["batch_size"]*len(trainloader)} | train_loss: {epoch_loss/len(trainloader):.4f}, lr: {optimizer.param_groups[0]["lr"]:.7f}'
        )


        if cfg["wandb"] == True:
            logger.info("Wandb logging")
            wandb.log(
                {
                    "Epoch": epoch,
                    "Train Loss (Epoch)": epoch_loss / len(trainloader),
                    "Learning Rate": optimizer.param_groups[0]["lr"],
                    "anticipation average mAP": mAP,
                    "anticipation average cAP": cAP,
                }
            )

    # os.rename(osp.join(result_path, 'ckpts', 'best.pth'), osp.join(result_path, 'ckpts', f'best_{best_mAP*100:.2f}.pth'))
    # save the training result and config as txt file
    with open(osp.join(result_path, "result.txt"), "w") as f:
        f.write(f"Best mAP: {best_mAP*100:.2f} at epoch {best_epoch}\n")
        f.write(f"Best cAP: {best_cAP*100:.2f} at epoch {best_cAP_epoch}\n")
        f.write(f"Config: {cfg}")

    wandb.finish() if args.wandb else None
