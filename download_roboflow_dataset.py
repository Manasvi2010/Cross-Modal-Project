from roboflow import Roboflow

rf = Roboflow(api_key="2ieVJZA8vNpkcPunq9HI")

project = rf.workspace("foodtest-es2zb").project("indian-food-dca77-psnrm")

dataset = project.version("latest").download("folder")

print("✅ Indian Food dataset downloaded successfully")