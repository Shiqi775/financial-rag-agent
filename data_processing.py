import os
import re
from config import ORIGINAL_DATA_DIR, PROCESSED_DATA_DIR
from bs4 import BeautifulSoup
import shutil
from tqdm import tqdm

# # TODO: take the downloaded data from ORIGINAL_DATA_DIR,
# # clean the documents (extract text information,
# # remove boilerplate, etc.) and save the cleaned data
# # to PROCESSED_DATA_DIR in the following format.

def process_10k_files():
    """Process and organize 10-K files from ORIGINAL_DATA_DIR to PROCESSED_DATA_DIR"""
    
    # Get list of tickers from ORIGINAL_DATA_DIR
    sec_edgar_path = os.path.join(ORIGINAL_DATA_DIR, "sec-edgar-filings")
    tickers = [d for d in os.listdir(sec_edgar_path) if os.path.isdir(os.path.join(sec_edgar_path, d))]
    
    # Create PROCESSED_DATA_DIR if it doesn't exist
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    for ticker in tqdm(tickers, desc="Processing tickers"):
        # Path to ticker's 10-K directory
        ticker_10k_path = os.path.join(sec_edgar_path, ticker, "10-K")
        
        # Get all filing folders
        filing_folders = os.listdir(ticker_10k_path)
        
        # Process each filing
        for folder in filing_folders:
            # Extract year from folder name (format: 000319201-19-000031)
            year_match = re.search(r'-(\d{2})-', folder)
            if year_match:
                year = "20" + year_match.group(1)
                
                # Create directory for this filing
                processed_dir = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_{year}")
                os.makedirs(processed_dir, exist_ok=True)
                
                # Path to full submission file
                submission_file = os.path.join(ticker_10k_path, folder, "full-submission.txt")
                
                try:
                    # Read the submission file
                    with open(submission_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Extract and clean the content
                    cleaned_content = clean_10k_content(content)
                    
                    # Save cleaned content
                    output_file = os.path.join(processed_dir, "content.txt")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(cleaned_content)
                        
                except Exception as e:
                    print(f"Error processing {ticker} {year}: {str(e)}")

def clean_10k_content(content):
    # Remove non-printable characters
    content = re.sub(r'[^\x20-\x7E\n]', '', content)
    
    # Extract main content
    content_parts = content.split('<', 1)
    if len(content_parts) > 1:
        content = '<' + content_parts[1]
    
    soup = BeautifulSoup(content, 'lxml')
    
    # 1. Remove standard noise elements
    for tag in soup(['script', 'style', 'meta', 'head', 'link']):
        tag.decompose()
    
    # 2. Remove SEC headers and footers 
    patterns_to_remove = [
        'UNITED STATES SECURITIES AND EXCHANGE COMMISSION',
        'FORM 10-K',
        r'EDGAR.*HEADER',
        'Commission file number',
        'CONFORMED SUBMISSION.*TYPE',
        'CONFORMED PERIOD.*REPORT'
    ]
    for pattern in patterns_to_remove:
        for element in soup.find_all(text=re.compile(pattern)):
            element.extract()
    
    # 3. Enhanced table processing 
    for table in soup.find_all('table'):
        # Only process tables with actual content
        if not table.get_text(strip=True):
            table.decompose()
            continue
            
        # Remove empty cells
        for cell in table.find_all(['td', 'th']):
            if not cell.get_text(strip=True):
                cell.decompose()
            
        # Clean cell formatting but preserve structure
        for tag in table.find_all(True):
            tag.attrs = {}  # Remove attributes but keep structure

    # 4. Improve paragraph handling
    for tag in soup.find_all(['p', 'div', 'tr']):
        # Add newline only if it doesn't end with one
        if not tag.get_text().strip().endswith('\n'):
            tag.append('\n')
    
    # 5. Preserve important section headers
    sections = []
    section_pattern = re.compile(r'(?:^|\n)(?:ITEM|Item)\s+\d+[A-Za-z]?\.(?:\s+|$)')
    for item_header in soup.find_all(text=section_pattern):
        sections.append(item_header.strip())
    
    # 6. Normalize whitespace
    text = soup.get_text(separator=' ')
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'\s+', ' ', text)
    
    # 7. Final cleanup
    text = re.sub(r'(?<=[.!?])\s+(?=[A-Z])', '\n', text)  # Split sentences into paragraphs
    text = re.sub(r'\n{3,}', '\n\n', text)  # Remove excessive newlines
    
    important_content = "\n".join(sections) + "\n\n" + text
    return important_content.strip()

process_10k_files()

# # In terms of implementation, this part is the most flexible.
# # The bare minimum for this part (valued 5 points) is saving
# # the original documents in the format described before
# # without processing them. The remaining 10 points will be given
# # for the attempts to clean the data.

# # There is no objective criteria for the quality of cleaning.
# # As the first thing to try, we would suggest removing
# # html tags. However, tables in this case become a mess,
# # and it is very likely that an LLM would do better if
# # the html structure was preserved.

# # Please make reasonable efforts to manipulate HTML tags effectively,
# # remove unnecessary boilerplate content, and explore other
# # preprocessing techniques as deemed appropriate.

# # Possible scores:
# # [10 pts]         The processed documents is stored
# #                 in the aforementioned format.
# # [5 pts]       The format for the processed data
# #                 is not preserved.
# # [up to +10 pts] Some level of cleaning is done.