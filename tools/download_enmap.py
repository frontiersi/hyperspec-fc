#!/usr/bin/env python3

import os
import requests
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import getpass

# ============================
# Logging Configuration
# ============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("download_enmap.log"),
        logging.StreamHandler()
    ]
)

class SessionWithRetry(requests.Session):
    """
    Custom session class with retry strategy.
    """
    def __init__(self, retries=3, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504)):
        super().__init__()
        retry_strategy = Retry(
            total=retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            allowed_methods=["GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.mount("https://", adapter)
        self.mount("http://", adapter)

def download_enmap(urls, username, password, download_folder=None):
    """
    Downloads multiple GeoTIFF files from an authenticated service using a single authentication session.

    Parameters:
    - urls (str or list of str): A single URL or a list of direct URLs to the GeoTIFF files.
    - download_folder (str): The local directory to save the downloaded files.
    - username (str): Your authentication username.
    - password (str): Your authentication password.

    Raises:
    - Exception: If any step in the authentication or download process fails.
    """
    # Convert single URL string to a list for uniform processing
    if isinstance(urls, str):
        urls = [urls]
    elif not isinstance(urls, list):
        logging.error("The 'urls' parameter must be a list of URLs or a single URL string.")
        raise ValueError("The 'urls' parameter must be a list of URLs or a single URL string.")

    # Set the save directory to current directory if not specified
    if download_folder is None:
        download_folder = os.getcwd()
    else:
        # Create the directory if it does not exist
        os.makedirs(download_folder, exist_ok=True)
    logging.info(f"Download folder set to: {download_folder}")

    # Initialize a session with retry strategy
    session = SessionWithRetry()

    # Define common headers to mimic a browser
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/133.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,image/apng,*/*;q=0.8,"
            "application/signed-exchange;v=b3;q=0.7"
        ),
        "Accept-Language": "en-AU,en;q=0.9",
    }

    try:
        # Step 1: Authenticate and obtain the ticket
        first_url = urls[0]
        login_service_url = first_url
        login_url = f"https://sso.eoc.dlr.de/eoc/auth/login?service={login_service_url}"
        headers["Referer"] = login_url

        logging.info(f"Initiating authentication with login URL: {login_url}")

        # GET the login page to retrieve hidden form fields
        response = session.get(login_url, headers=headers)
        response.raise_for_status()
        logging.info("Successfully retrieved the login page.")

        # Parse the login page HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the login form
        login_form = soup.find('form')
        if not login_form:
            logging.error("Login form not found on the login page.")
            raise Exception("Login form not found on the login page.")

        # Extract all form inputs
        form_data = {}
        for input_tag in login_form.find_all('input'):
            input_name = input_tag.get('name')
            input_value = input_tag.get('value', '')
            if input_name:
                form_data[input_name] = input_value

        # Update form data with username and password
        form_data.update({
            'username': username,
            'password': password,
            '_eventId': 'submit',         # Typically required for form submission
            'geolocation': '',            # Assuming geolocation is optional/empty
        })

        logging.info("Submitting login credentials.")
        # POST the login form with credentials
        post_response = session.post(login_url, data=form_data, headers=headers, allow_redirects=False)
        post_response.raise_for_status()

        # Check for redirection to get the ticket
        if post_response.status_code != 302:
            logging.error("Login failed or unexpected response received.")
            raise Exception("Login failed or unexpected response received.")

        # Extract the 'Location' header to get the redirected URL with the ticket
        redirect_url = post_response.headers.get('Location')
        if not redirect_url:
            logging.error("Redirection URL not found after login.")
            raise Exception("Redirection URL not found after login.")

        # Parse the redirected URL to extract the ticket parameter
        parsed_redirect = urlparse(redirect_url)
        query_params = parse_qs(parsed_redirect.query)
        ticket = query_params.get('ticket', [None])[0]
        if not ticket:
            logging.error("Authentication ticket not found in the redirected URL.")
            raise Exception("Authentication ticket not found in the redirected URL.")

        logging.info(f"Authentication successful. Ticket obtained: {ticket}")

        # Iterate through the list of URLs and download each file
        for idx, url in enumerate(urls, start=1):
            logging.info(f"Processing URL ({idx}/{len(urls)}): {url}")

            # Parse the download URL to check if ticket is already present
            parsed_url = urlparse(url)
            download_query = parse_qs(parsed_url.query)

            # If 'ticket' is not present, append it
            if 'ticket' not in download_query:
                download_query['ticket'] = ticket
                new_query = urlencode(download_query, doseq=True)
                download_url_with_ticket = urlunparse(parsed_url._replace(query=new_query))
                logging.debug(f"Appended ticket to URL: {download_url_with_ticket}")
            else:
                download_url_with_ticket = url

            # Update headers for the download request
            download_headers = headers.copy()
            download_headers["Referer"] = "https://sso.eoc.dlr.de/"

            try:
                # Send the GET request to download the file
                with session.get(download_url_with_ticket, headers=download_headers, stream=True, timeout=(15, 60)) as download_response:
                    download_response.raise_for_status()
                    logging.info(f"Started downloading: {url}")

                    # Extract the filename from the URL
                    filename = os.path.basename(parsed_url.path)
                    if not filename:
                        filename = f"downloaded_file_{idx}.tif"  # Fallback filename

                    file_path = os.path.join(download_folder, filename)

                    # Get the total file size for the progress bar
                    total_size_in_bytes = int(download_response.headers.get('content-length', 0))
                    block_size = 1024 * 1024  # 1MB

                    # Initialize the progress bar
                    progress_bar = tqdm(
                        total=total_size_in_bytes, 
                        unit='iB', 
                        unit_scale=True, 
                        desc=filename,
                        leave=True
                    )

                    # Write the content to a file in chunks
                    with open(file_path, 'wb') as file:
                        for chunk in download_response.iter_content(chunk_size=block_size):
                            if chunk:
                                file.write(chunk)
                                progress_bar.update(len(chunk))
                    progress_bar.close()

                    # Verify if the download completed successfully
                    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                        logging.warning(f"Mismatch in expected size for '{filename}'. Download may be incomplete.")
                    else:
                        logging.info(f"Successfully downloaded '{filename}' to '{download_folder}'.")

            except requests.exceptions.HTTPError as download_http_err:
                logging.error(f"HTTP error occurred while downloading '{url}': {download_http_err}")
            except requests.exceptions.RequestException as download_req_err:
                logging.error(f"Request exception occurred while downloading '{url}': {download_req_err}")
            except Exception as download_err:
                logging.error(f"An unexpected error occurred while downloading '{url}': {download_err}")

    except requests.exceptions.RequestException as auth_err:
        logging.error(f"An error occurred during the authentication request: {auth_err}")
        raise Exception(f"An error occurred during the authentication request: {auth_err}")
    except Exception as ex:
        logging.error(f"An error occurred: {ex}")
        raise Exception(f"An error occurred: {ex}")

   
if __name__ == "__main__":
    import sys

    # Usage is python download_emit.py url_file save_dir username password
    if len(sys.argv) < 2:
        print("Usage: python download_enmap.py url_file")
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

    # EnMap credentials
    username = getpass.getpass("Enter your EnMap username: ")
    password = getpass.getpass("Enter your EnMap password: ")

    # Call the download function
    download_enmap(urls, username, password, save_dir)