import copy

import torch

from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
from utils.logger import WandbLogger, BasicLogger


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):
    _set_random(args["seed"])
    _set_device(args)

    if args["logger"] == "wandb":
        logger = WandbLogger(args)
    elif args["logger"] == "basic":
        logger = BasicLogger(args)
    else:
        raise ValueError("Invalid logger type.")

    logger.print_args()

    data_manager = DataManager(
        dataset_name=args["dataset"],
        shuffle=args["shuffle"],
        seed=args["seed"],
        init_cls=args["init_cls"],
        increment=args["increment"],
        args=args,
    )

    args["nb_classes"] = data_manager.nb_classes  # update args
    args["nb_tasks"] = data_manager.nb_tasks
    model = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    for task in range(data_manager.nb_tasks):
        logger.info("All params: {}".format(count_parameters(model._network)))
        logger.info(
            "Trainable params: {}".format(
                count_parameters(model._network, True))
        )
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()

        if nme_accy is not None:
            logger.info("CNN: {}".format(cnn_accy["grouped"]))
            logger.info("NME: {}".format(nme_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])

            data = {"class": (task + 1) * args["increment"],
                    "cnn_top1": cnn_accy["top1"], "cnn_top5": cnn_accy["top5"],
                    "nme_top1": nme_accy["top1"], "nme_top5": nme_accy["top5"],
                    "cnn_average_acc": sum(cnn_curve["top1"]) / len(cnn_curve["top1"]),
                    "nme_average_acc": sum(nme_curve["top1"]) / len(nme_curve["top1"])
                    }

            logger.log(data)
            logger.info(f"Average Accuracy (CNN): {data['cnn_average_acc']}")
            logger.info(f"Average Accuracy (NME): {data['nme_average_acc']}")

        else:
            logger.info("No NME accuracy.")
            logger.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            data = {"class": (task + 1) * args["increment"],
                    "cnn_top1": cnn_accy["top1"], "cnn_top5": cnn_accy["top5"],
                    "cnn_average_acc": sum(cnn_curve["top1"]) / len(cnn_curve["top1"]),
                    }

            logger.log(data)
            logger.info(f"Average Accuracy (CNN): {data['cnn_average_acc']}")


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
