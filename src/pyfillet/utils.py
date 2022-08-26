import hashlib
from tqdm import tqdm


def md5(filename, blocksize=4096):
    """Compute MD5 of the file."""
    md5_hash = hashlib.md5()
    with open(filename, "rb") as fp:
        for byte_block in iter(lambda: fp.read(blocksize), b""):
            md5_hash.update(byte_block)
    return md5_hash.hexdigest()


class DownloadBar:
    """Progress bar hook for URL downloading."""
    def __init__(self):
        self._pbar = tqdm()

    def __call__(self, chunk, max_size, total_size):
        percent = int(100 * min(chunk * max_size / total_size,  1))
        self._pbar.set_description("Downloaded {}%".format(percent))

    def close(self):
        self._pbar.close()
