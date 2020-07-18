import os
import argparse
import sys
import glob
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--faildir', help='Name of the folder containing images to perform fail detection on.',
                    default=None)
args = parser.parse_args()
FA_DIR = args.faildir
CWD_PATH = os.getcwd()
if FA_DIR:
    PATH_TO_FAIMAGES = os.path.join(CWD_PATH,FA_DIR)
    failimages = glob.glob(PATH_TO_FAIMAGES)[0]


def generatefakefile(PATH):
    with open(PATH+'\\'+"fail.txt", 'w') as file:
        file.write('fail')
    print('fakefile')

generatefakefile(failimages)