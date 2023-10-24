from config.configurator import configs
import importlib

def build_data_handler():
    datahandler_name = 'data_handler_' + configs['data']['type']
    module_path = ".".join(['data_utils', datahandler_name])
    if importlib.util.find_spec(module_path) is None:
        raise NotImplementedError('DataHandler {} is not implemented'.format(datahandler_name))
    module = importlib.import_module(module_path)
    for attr in dir(module):
        if attr.lower() == datahandler_name.lower().replace('_', ''):
            return getattr(module, attr)()
    else:
        raise NotImplementedError('DataHandler Class {} is not defined in {}'.format(datahandler_name, module_path))



