#!/usr/bin/env python3

import os
import requests
import logging
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from tqdm import tqdm
import getpass

# ============================
# Logging Configuration
# ============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("download_emit.log"),
        logging.StreamHandler()
    ]
)

class SessionWithHeaderRedirection(requests.Session):
    """
    Custom session class to handle authentication and maintain headers during redirects.
    """
    AUTH_HOST = 'urs.earthdata.nasa.gov'

    def __init__(self, username, password):
        super().__init__()
        self.auth = (username, password)

    def rebuild_auth(self, prepared_request, response):
        """
        Overrides the default rebuild_auth to maintain headers when redirected to or from the authentication host.
        """
        headers = prepared_request.headers
        url = prepared_request.url

        if 'Authorization' in headers:
            original_parsed = requests.utils.urlparse(response.request.url)
            redirect_parsed = requests.utils.urlparse(url)

            if (original_parsed.hostname != redirect_parsed.hostname) and \
               (redirect_parsed.hostname != self.AUTH_HOST) and \
               (original_parsed.hostname != self.AUTH_HOST):
                del headers['Authorization']

        return

def setup_session(username, password):
    """
    Sets up the session with retry strategy.

    Parameters:
        username (str): Earthdata Login username.
        password (str): Earthdata Login password.

    Returns:
        requests.Session: Configured session object.
    """
    session = SessionWithHeaderRedirection(username, password)
    retry_strategy = Retry(
        total=3,  # Total number of retries
        status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry
        allowed_methods=["GET", "POST"],  # HTTP methods to retry
        backoff_factor=1  # A backoff factor for sleep between retries
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def download_emit(urls, username, password, save_dir=None):
    """
    Downloads one or multiple files from given URLs using Earthdata Login credentials.

    Parameters:
        urls (str or list of str): URL(s) of the files to download.
        username (str): Your Earthdata Login username.
        password (str): Your Earthdata Login password.
        save_dir (str, optional): Directory to save the downloaded files. Defaults to current directory.

    Raises:
        requests.exceptions.HTTPError: If an HTTP error occurs during any download.
        Exception: For any other exceptions that may occur.
    """
    # Convert single URL string to a list for uniform processing
    if isinstance(urls, str):
        urls = [urls]
    elif not isinstance(urls, list):
        raise ValueError("The 'urls' parameter must be a list of URLs or a single URL string.")

    # Set the save directory to current directory if not specified
    if save_dir is None:
        save_dir = os.getcwd()
    else:
        # Create the directory if it does not exist
        os.makedirs(save_dir, exist_ok=True)

    # Initialize the custom session with credentials and retry strategy
    session = setup_session(username, password)

    # Iterate through each URL and download the file
    for idx, url in enumerate(urls, start=1):
        # Extract the filename from the URL
        filename = os.path.basename(url)
        if not filename:
            filename = f"downloaded_file_{idx}.tif"  # Fallback filename if extraction fails
        save_path = os.path.join(save_dir, filename)

        try:
            # Submit the request using the session
            with session.get(url, stream=True) as response:
                logging.info(f"\nRequesting URL ({idx}/{len(urls)}): {url}")
                logging.info(f"HTTP Status Code: {response.status_code}")

                # Raise an exception in case of HTTP errors
                response.raise_for_status()

                # Get the total file size for the progress bar
                total_size_in_bytes = int(response.headers.get('content-length', 0))
                block_size = 1024 * 1024  # 1MB

                # Initialize the progress bar
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc=filename)

                # Save the file in chunks with progress update
                with open(save_path, 'wb') as fd:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:  # Filter out keep-alive chunks
                            fd.write(chunk)
                            progress_bar.update(len(chunk))
                progress_bar.close()

                # Verify if the download completed successfully
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    logging.warning(f"Mismatch in expected size for '{filename}'. Download may be incomplete.")
                else:
                    logging.info(f"Successfully downloaded '{filename}' to '{save_dir}'.")

        except requests.exceptions.HTTPError as http_err:
            logging.error(f"HTTP error occurred while downloading '{filename}': {http_err}")
        except requests.exceptions.RequestException as req_err:
            logging.error(f"Request exception occurred while downloading '{filename}': {req_err}")
        except Exception as err:
            logging.error(f"An unexpected error occurred while downloading '{filename}': {err}")


if __name__ == "__main__":
    import sys

    # Usage is python download_emit.py url_file
    if len(sys.argv) < 2:
        print("Usage: python download_emit.py url_file")
        sys.exit(1)
    
    # Read the URLs from the file
    url_file = sys.argv[1]
    with open(url_file, 'r') as f:
        urls = f.readlines()
    urls = [url.strip() for url in urls]

    # Save directory is the same as the url file
    save_dir = os.path.dirname(url_file)
    if not save_dir:
        save_dir = os.getcwd()

    # Earthdata credentials
    username = getpass.getpass("Enter your EarthData username: ")
    password = getpass.getpass("Enter your EarthData password: ")

    # Call the download function
    download_emit(urls, username, password, save_dir)



