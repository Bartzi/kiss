import configparser


def parse_config(file_name, args):
    config = configparser.ConfigParser()
    config.read(file_name)

    for key in config['PATHS']:
        value = config['PATHS'][key]
        if len(value) == 0:
            value = None
        setattr(args, key, value)

    for key in config["SIZES"]:
        values = [int(v.strip()) for v in config['SIZES'][key].split(',')]
        setattr(args, key, values)

    for section, conversion_func in zip(["HYPERPARAMETERS_FLOAT", "HYPERPARAMETERS_INT"], [float, int]):
        for key in config[section]:
            setattr(args, key, conversion_func(config[section][key]))

    for key in config["TRAIN_PARAMS"]:
        setattr(args, key, int(config["TRAIN_PARAMS"][key]))

    for key in config["DATASET_SETTINGS"]:
        setattr(args, key, config["DATASET_SETTINGS"][key] == "True")

    for key in config["TEST_DATASETS"]:
        setattr(args, f"test_dataset_{key}", config["TEST_DATASETS"][key])

    if "TEST_DATASETS_ALNUM_ONLY" in config:
        for key in config["TEST_DATASETS_ALNUM_ONLY"]:
            setattr(args, f"test_alnum_only_{key}", config["TEST_DATASETS_ALNUM_ONLY"][key] == "True")

    return args
