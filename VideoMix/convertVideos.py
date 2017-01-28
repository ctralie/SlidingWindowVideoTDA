import subprocess
import os

if __name__ == '__main__':
    for i in range(599):
        subprocess.call(["avconv", "-i", "KTH5SecondOggs/%i.ogg"%i, "-b:v", "30000k", "KTH5SecondOggs/%i.webm"%i])
