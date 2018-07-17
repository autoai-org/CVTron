from cvtron.data_zoo.compress_util import ArchiveFile
from cvtron.data_zoo.compress_util import ToArchiveFolder

def compress(folderPath):
    taf = ToArchiveFolder(folderPath)
    taf.zip('../folder.zip')

def uncompress():
    pass

if __name__ == '__main__':
    compress('./')