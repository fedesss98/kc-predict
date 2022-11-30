import neptune.new as neptune
import json

from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent.parent.parent

def log_neptune():
    run = neptune.init(
        project="unipa-it-ml/ml-to-citrus-orchard-kc",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3YzY1ZDA4NS02Mzk4LTRhMTktYjg4Yy1kZDk5YTM0ZDJjY2MifQ==",
    )  # Service account credentials
    
    # Open JSON file
    with open(ROOT_DIR/'docs'/'run.json', "r") as f:
        data = json.load(f)
        run["sys/tags"].add(data["tags"])
        run["user"] = data["user"]
        run["model/name"] = data["model"]["name"]
        run["model/parameters"] = data["model"]["parameters"]
        run["train/score"] = data["train_score"]
        run["test/score"] = data["test_score"]
    print(run)
    run.stop()
    return None


def start_neptune():
    pass


if __name__ == "__main__":
    log_neptune()

