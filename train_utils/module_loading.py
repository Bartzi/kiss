import importlib
import os


def load_module(module_file, module_path="models.model"):
    module_spec = importlib.util.spec_from_file_location(module_path, module_file)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def get_class(class_name, module_name, base_dir):
    module = load_module(os.path.abspath(os.path.join(base_dir, module_name)))
    klass = eval('module.{}'.format(class_name))
    return klass
