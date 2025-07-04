import pandas as pd
import phoenix as px
import yaml
from dotenv import load_dotenv

load_dotenv()

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

dataset_config = config["two_stage_dataset"]

df = pd.read_csv(dataset_config["path"])

phoenix_client = px.Client()

dataset = phoenix_client.upload_dataset(
    dataframe=df,
    dataset_name=dataset_config["dataset_name"],
    input_keys=dataset_config["input_field"],
    output_keys=dataset_config["output_field"],
    metadata_keys=dataset_config["metadata_field"],
)
