from torchvision.datasets.utils import download_and_extract_archive, download_url


import os
from typing import Optional


def download_file(url: str,
                  download_root: str,
                  extract_root: Optional[str] = None,
                  filename: Optional[str] = None,
                  extract=False) -> None:
    """
    Download a file from a url and optionally extract it to a target directory.
    Args:
        url (str): URL to download file from
        download_root (str): Directory to place downloaded file in
        extract_root (str, optional): Directory to extract downloaded file to
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        extract (bool, optional): If True, extract the downloaded file. Otherwise, do not extract.
    """
    if not extract:  # if extract is False
        download_root = os.path.expanduser(download_root)  # expand the user's home directory in download_root
        if not filename:  # if filename is not provided
            filename = os.path.basename(url)  # set filename to the basename of the URL

        download_url(url, download_root, filename, md5=None)  # download the file
    else:
        if extract_root is None:  # if extract_root is not provided
            extract_root = download_root  # set extract_root to download_root
        download_and_extract_archive(url, download_root, extract_root, filename=filename)  # download and extract the file to extract_root