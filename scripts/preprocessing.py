import hydra
import importlib

@hydra.main(config_path='../configs', config_name='config', version_base=None)
def main(cfg):
    preprocessor_path = cfg.preprocessor
    
    pkg = '.'.join(preprocessor_path.split('.')[:-1])
    class_name = preprocessor_path.split('.')[-1]
    
    preprocessor_class = getattr(importlib.import_module(pkg), class_name)
    preprocessor = preprocessor_class(cfg.datapath)
    preprocessor.process()
    
if __name__ == '__main__':
    main()