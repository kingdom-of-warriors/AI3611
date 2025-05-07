from huggingface_hub import HfApi
api = HfApi()

folder_path = "/home/ljr/AI3611/img_cap/data1"
repo_id = "jiarui1/flickr"
api.upload_large_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type="dataset",  # 或 "model", "space"
)
print(f"文件夹已上传到 {repo_id}")