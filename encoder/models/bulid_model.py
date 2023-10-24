from config.configurator import configs
import importlib

def build_model(data_handler):
    model_type = configs['data']['type']
    model_name = configs['model']['name']
    module_path = ".".join(['models', model_type, model_name])
    if importlib.util.find_spec(module_path) is None:
        raise NotImplementedError('Model {} is not implemented'.format(model_name))
    module = importlib.import_module(module_path)
    for attr in dir(module):
        if attr.lower() == model_name.lower():
            return getattr(module, attr)(data_handler)
    else:
        raise NotImplementedError('Model Class {} is not defined in {}'.format(model_name, module_path))



