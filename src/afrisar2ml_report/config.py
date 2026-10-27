from afrisar2ml_report import sites

# dataset config

DS_VERSION = 100  # version of the AfriSAR-2-ML dataset


# benchmark config

REPORT_VERSION = 100  # version of the report benchmark code
TRAIN_VAL_SITES = [sites.LOPE, sites.MABOUNIE, sites.MITZIC, sites.PONGARA]
TEST_SITE = sites.MONDAH
SEED = 21  # seed for the rng
N_RUNS = 5  # number of model training runs, different seeds
