import gdown

url = "https://drive.usercontent.google.com/download?id=1RPYTREPZ_PVzOHnESgTdg8x7MjVMl_4c&export=download&authuser=0&confirm=t&uuid=e6148024-cd81-4ddf-a2bc-a7c2da29a55a"
output = "3-gram.arpa"
gdown.download(url, output, quiet=False)

url = "https://drive.usercontent.google.com/download?id=1NF7tKEBjIsLYqd2TtMSx6R9hr-mMTYGK&export=download&authuser=0&confirm=t&uuid=b4fb86ad-7118-4a92-b51c-42b86c178b1e"
output = "librispeech-vocab.txt"
gdown.download(url, output, quiet=False)