import hydra
import importlib
import os
import sys

from pathlib import Path

root_dir = Path(__file__).parent.parent.resolve()
os.environ['PYTHONPATH'] = str(root_dir)
sys.path.insert(0, str(root_dir))

@hydra.main(config_path='../configs', config_name='config', version_base=None)
def main(cfg):
    trainer_path = cfg.trainer

    pkg = '.'.join(trainer_path.split('.')[:-1])
    class_name = trainer_path.split('.')[-1]

    trainer_class = getattr(importlib.import_module(pkg), class_name)
    trainer = trainer_class(cfg)
    trainer.run()

if __name__ == '__main__':
    main()