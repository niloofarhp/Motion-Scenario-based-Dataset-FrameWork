import os
import hydra
from omegaconf import DictConfig, OmegaConf
import evaluate.test_flow as evtf


@hydra.main(version_base=None, config_path="conf_eval", config_name="config")
def eval_script(cfg: DictConfig) -> None:
    # Sanity check
    print("Evaluation starting...")
    print("Sanity check OmegaConf.to_yaml(cfg):\n", OmegaConf.to_yaml(cfg))

    # Get simple dictionary
    # dictArgs = OmegaConf.to_container(cfg, resolve=True)
    # if cfg.error.relative :
    #     print('yes')
    #     print(cfg['error']['relative'])
    # else:
    #      print('no')F
    # print(cfg.tfds_config)

    # Ensure the base path is correctly formed
    dataset_base_path = os.path.join(cfg.base, cfg.tfds)
    print(f"Loading datasets in {str(dataset_base_path)} ...")

    if not os.path.exists(dataset_base_path):
        print("The specified dataset base path does not exist: " + str(dataset_base_path))
        return

    for dataset_folder in os.listdir(str(dataset_base_path)):
        if not dataset_folder.startswith('.'):
            print('\n', '=' * 30)
            print(f"\n[eval_script] Processing dataset: {dataset_folder}\n")
            cfg.tfds_config = dataset_folder
            evtf.TestFlow(cfg).run()

    evtf.TestFlow(cfg).run()


if __name__ == "__main__":
    eval_script()
