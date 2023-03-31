from .different_dataset import _C as DDC

config_factory = {
    'different_dataset': DDC,
}

def build_config(factory):
    return config_factory[factory]