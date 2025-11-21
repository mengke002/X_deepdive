"""
Utility module for data management and file handling.
"""
import os
import zipfile
import logging

logger = logging.getLogger(__name__)

def ensure_data_ready(followers_dir='X_followers', replies_dir='X_replies'):
    """
    Ensure data directories exist. If not, try to unzip from archive files in the project root.

    Args:
        followers_dir (str): Directory name for followers data.
        replies_dir (str): Directory name for replies data.
    """
    root_dir = os.getcwd() # Assuming script is run from root

    # Define expected zip paths
    followers_zip = os.path.join(root_dir, f"{followers_dir}.zip")
    replies_zip = os.path.join(root_dir, f"{replies_dir}.zip")

    # Check and extract Followers
    if not os.path.exists(followers_dir):
        logger.info(f"Directory '{followers_dir}' not found. Checking for zip...")
        if os.path.exists(followers_zip):
            logger.info(f"Extracting {followers_zip}...")
            try:
                with zipfile.ZipFile(followers_zip, 'r') as zip_ref:
                    zip_ref.extractall(root_dir)
                logger.info(f"Successfully extracted to {followers_dir}/")
            except Exception as e:
                logger.error(f"Failed to extract {followers_zip}: {e}")
        else:
            logger.warning(f"Neither directory '{followers_dir}' nor zip file found.")
    else:
        logger.info(f"Directory '{followers_dir}' exists.")

    # Check and extract Replies
    if not os.path.exists(replies_dir):
        logger.info(f"Directory '{replies_dir}' not found. Checking for zip...")
        if os.path.exists(replies_zip):
            logger.info(f"Extracting {replies_zip}...")
            try:
                with zipfile.ZipFile(replies_zip, 'r') as zip_ref:
                    zip_ref.extractall(root_dir)
                logger.info(f"Successfully extracted to {replies_dir}/")
            except Exception as e:
                logger.error(f"Failed to extract {replies_zip}: {e}")
        else:
            logger.warning(f"Neither directory '{replies_dir}' nor zip file found.")
    else:
        logger.info(f"Directory '{replies_dir}' exists.")
