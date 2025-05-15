import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="config")
def parseworker(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))  # print the configuration for debugging

    dict_args = OmegaConf.to_container(cfg, resolve=True)  # convert the configuration to a dictionary

    # Build the argument list, ensuring all values are converted to strings
    list_args = []
    for k, v in dict_args.items():
        if k == "job_dir":
            k = "job-dir"
        if isinstance(v, bool):
            if v:  # only add the flag if it's true
                list_args.append(f"--{k}")
        else:
            list_args.append(f"--{k}")  # append the argument key
            list_args.append(str(v))  # append the argument value, ensuring it's a string

    to_worker = ["/usr/bin/python3", "scenes/movi.py"] + list_args
    print("\nHydra args:", to_worker)

    import subprocess
    subprocess.run(to_worker)


if __name__ == "__main__":
    parseworker()
