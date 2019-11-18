import os
import sys

from types import ModuleType

from train_utils.module_loading import load_module

# we assume that our code is always run from the root dir of this repo and nobody tampers with the python path
# we use this to determine whether we should make a backup of the file of a class or not, because if it is from
# our code base it might contain breaking changes in the future
code_root_path = sys.path[0]


def get_import_info(chain):
    backup_info = get_backup_info(get_module(chain), chain)
    return {
        "import_string": f"from {backup_info['module_path']} import {backup_info['class_name']}",
        "class_name": backup_info['class_name'],
    }


def get_backup_info(module, obj):
    return {
        "module_path": module.__name__,
        "class_name": obj.__class__.__name__,
        "files": (get_definition_filepath(obj), get_definition_filename(obj))
    }


def get_module(obj):
    return __import__(obj.__module__, fromlist=obj.__module__.split('.')[:-1])


def get_definition_filepath(obj):
    return get_module(obj).__file__


def get_definition_filename(obj):
    return os.path.basename(get_definition_filepath(obj))


def restore_backup(backup_plan, backup_dir):
    if 'import_string' in backup_plan:
        exec(backup_plan['import_string'])
        klass = eval(backup_plan['class_name'])
        return klass

    # 1. load root module with network definition
    root_module = load_module(os.path.abspath(os.path.join(backup_dir, backup_plan['root_object']["files"][1])))

    # 2. adapt pointers to other modules that we created a backup of
    for child in backup_plan['children']:
        child_module = load_module(
            os.path.abspath(os.path.join(backup_dir, child['files'][1])),
            module_path=child['module_path']
        )
        for name in filter(lambda x: not x.startswith('_'), dir(root_module)):
            module_attr = getattr(root_module, name)
            if isinstance(module_attr, ModuleType) or module_attr.__module__ != child['module_path']:
                # if the attr we grabbed from the module is itself a module (so make sure to import everything directly!!)
                # or if the module path does not fit to the path we saved we have a look at the next attr
                continue

            if module_attr.__name__ != child['class_name']:
                # we do not have the correct class right now
                continue

            setattr(root_module, name, getattr(child_module, child['class_name']))
            break

    return getattr(root_module, backup_plan['root_object']['class_name'])
