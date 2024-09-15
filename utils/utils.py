def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def save_config(config, save_path):
    with open(save_path, "w") as file:
        yaml.dump(config, file)


def get_model(config):
    if config["name"] == "HRNet":
        from models.HRNet.hrnet import PoseHighResolutionNet

        model = PoseHighResolutionNet(config["config"])
        if config["weights"]:
            model.init_weights(config["weights"])
    else:
        raise ValueError(
            f"Model {config['model']['name']} not recognized"
        )  # Added error handling
    return model
