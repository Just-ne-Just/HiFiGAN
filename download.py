import gdown

url = "https://drive.usercontent.google.com/download?id=1YfZVcVBMZIlE2yw1Mr7sB5ILOCLG9-DW&export=download&authuser=0&confirm=t&uuid=3e64328b-71f3-4c18-879d-eeac522c7a61"
output = "model_best.pth"
gdown.download(url, output, quiet=False)
