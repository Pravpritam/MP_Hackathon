from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="AmineSam/irail-crowd-counting-yolov8n", filename="best.pt")

print("-"*50)
print(path)