import os
import shutil
import random
from sec_edgar_downloader import Downloader
from config import ORIGINAL_DATA_DIR
from sample_tickers import get_sp500_tickers_wikipedia

# Constants
EMAIL_ADDRESS = "siki_hu@outlook.com"
NUM_DOCS = 10   # Number of complete 10-K filings required per ticker
UNIQUE_ID = 904061372

# Get the current working directory
CURRENT_DIR = os.getcwd()

# Load the initial list of tickers from a file
with open("sampled_tickers.txt", "r") as f:
    sampled_tickers = [line.strip() for line in f.readlines()]

all_sp500_tickers = get_sp500_tickers_wikipedia()

# Initialize the Downloader
dl = Downloader(CURRENT_DIR, email_address=EMAIL_ADDRESS)

def clear_filing_folder(ticker):
    """
    Clears the entire 'sec-edgar-filings' folder.
    """
    filings_dir = os.path.join(CURRENT_DIR, "sec-edgar-filings")
    if os.path.exists(filings_dir):
        shutil.rmtree(filings_dir)
        # print(f"Cleared existing folder for {ticker}.")

def count_filing_folders(ticker):
    """
    Counts the number of valid 10-K filing folders for a specific ticker.
    """
    filings_dir = os.path.join("D:\8803_D\A13", "sec-edgar-filings", ticker, "10-K")
    if os.path.exists(filings_dir):
        return len([d for d in os.listdir(filings_dir) if os.path.isdir(os.path.join(filings_dir, d))])
    return 0

def resample_tickers(current_tickers):
    """
    Resamples tickers from the unprocessed ticker list.
    """
    global UNIQUE_ID
    UNIQUE_ID += 1
    random.seed(UNIQUE_ID)
    remaining_tickers = list(set(all_sp500_tickers) - set(current_tickers))
    return random.sample(remaining_tickers, NUM_DOCS)

def process_tickers(tickers):
    """
    Core logic: downloading, validation, and resampling.
    """
    valid_tickers = []
    for ticker in tickers:
        # print(f"Processing ticker: {ticker}")
        
        # Download 10-K filings
        dl.get("10-K", ticker, after="2010-01-01", before="2020-01-01")
        
        # Validate the completeness of historical filings
        num_filings = count_filing_folders(ticker)
        # print(f"Number of filings for {ticker}: {num_filings}")
        
        if num_filings == 10:
            valid_tickers.append(ticker)
            # print(f"Ticker {ticker} is valid.")
        else:
            # print(f"Ticker {ticker} is invalid. Clearing and resampling...")
            clear_filing_folder(ticker)
            new_tickers = resample_tickers(tickers)
            with open("sampled_tickers.txt", "w") as f:
                    f.write("\n".join(new_tickers))
            return process_tickers(new_tickers)  # Restart processing
        
        if len(valid_tickers) == NUM_DOCS:
            break
    
    return valid_tickers

# Start processing
sampled_tickers = process_tickers(sampled_tickers)

# Move 'sec-edgar-filings' directory to the target ORIGINAL_DATA_DIR
src_dir = os.path.join(CURRENT_DIR, "sec-edgar-filings")  # Original location of 'sec-edgar-filings'
dest_dir = os.path.join(ORIGINAL_DATA_DIR, "sec-edgar-filings")  # Target directory
if os.path.exists(src_dir):
    # Remove destination directory if it exists to avoid conflicts
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    
    # Move the entire directory
    shutil.move(src_dir, dest_dir)
    # print(f"Moved 'sec-edgar-filings' to {dest_dir}.")


# Output results
# print(f"Valid tickers with complete 10-year history: {sampled_tickers}")
