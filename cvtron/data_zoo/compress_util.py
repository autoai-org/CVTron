import os
import zipfile
from tqdm import tqdm
from cvtron.utils.logger.Logger import logger

class ArchiveFile:
    def __init__(self, filepath):
        if zipfile.is_zipfile(filepath):
            self._af = zipfile.ZipFile(filepath, 'r')
            self.filepath = filepath
            logger.info("Successfully load file" + filepath)
        else:
            logger.warn("Cannot load file" + filepath)
    def getInfo(self):
        return self._af.infolist()
    def unzip(self, extractTo, deleteOrigin=False):
        uncompress_size = sum((file.file_size for file in self._af.infolist()))
        extracted_size = 0
        pbar = tqdm(total=uncompress_size, initial=extracted_size, unit='B', unit_scale=True, desc="uncompressing "+self.filepath)
        for file in self._af.infolist():
            self._af.extract(file, extractTo)
            extracted_size += file.file_size
            pbar.update(extracted_size)
        self._af.extractall(extractTo)
        logger.info("Successfully unzip file")
        if deleteOrigin:
            os.remove(self.filepath)
            logger.info("Successfully delete original file")