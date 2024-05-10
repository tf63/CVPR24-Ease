import os
import sys
import logging
import wandb


class Logger:
    def __init__(self, args):
        # Prepare variables
        self.init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
        self.logs_name = os.path.join("logs",
                                      args["model_name"],
                                      args["dataset"],
                                      f"{self.init_cls}",
                                      args['increment']
                                      )

        self.logfilename = os.path.join("logs",
                                        args["model_name"],
                                        args["dataset"],
                                        f"{self.init_cls}",
                                        args["increment"],
                                        f"{args['prefix']}_{args['seed']}_{args['backbone_type']}"
                                        )
        self.args = args

    def print_args(self):
        for key, value in self.args.items():
            logging.info("{}: {}".format(key, value))

    def log(self, data):
        raise NotImplementedError("log method is not implemented.")

    def info(self, message):
        raise NotImplementedError("log method is not implemented.")


class BasicLogger(Logger):
    def __init__(self, args):
        super().__init__(args)

        if not os.path.exists(self.logs_name):
            os.makedirs(self.logs_name)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(filename)s] => %(message)s",
            handlers=[
                logging.FileHandler(filename=self.logfilename + ".log"),
                logging.StreamHandler(sys.stdout),
            ],
        )

    def log(self, data):
        logging.info("")
        pass

    def info(self, message):
        logging.info(message)


class WandbLogger(Logger):
    def __init__(self, args):
        super().__init__(args)

        wandb.init(project="cil-ease", name=self.logs_name, config=args)

    def log(self, data):
        wandb.log(data)

    def info(self, message):
        logging.info(message)