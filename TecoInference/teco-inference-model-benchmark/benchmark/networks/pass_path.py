import json
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent

def pass_path(model_name="", bs="default"):
    #bs is single card batch size.
    with open(ROOT_PATH / "pass/pass_map.json", 'r', encoding='utf-8') as f:
        pass_map = json.load(f)

    if (model_name in pass_map):
        if (bs in pass_map[model_name]):
            pass_path = pass_map[model_name][bs]
        else:
            pass_path = pass_map[model_name]["default"]
        pass_path = str(ROOT_PATH / pass_path)
        print("\033[32mThe model pass path is: \033[0m" + "\033[32m" + pass_path + "\033[0m")
    else:
        print("\033[0;33mThe model does not have pass file, will use default pass.\033[0m")
        return str(ROOT_PATH / "pass/default_pass.py")

    return pass_path
