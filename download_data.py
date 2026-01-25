import requests
import os

DATA_DIR = "ecommerce_analytics/data"
URLS = {
    "EcommerceDataset1.xlsx": "https://prod-files-secure.s3.us-west-2.amazonaws.com/d1e1bc70-9ede-4c69-84fd-42c5605803a0/07ec29ab-411b-4b17-a413-27f1fbd798c7/EcommerceDataset1.xlsx",
    "EcommerceDataset2.xlsx": "https://prod-files-secure.s3.us-west-2.amazonaws.com/d1e1bc70-9ede-4c69-84fd-42c5605803a0/edae67de-d62e-4613-b11a-2749b0a2149b/EcommerceDataset2.xlsx"
}

def download_file(filename, url):
    print(f"Downloading {filename}...")
    try:
        response = requests.get(url, allow_redirects=True)
        response.raise_for_status()
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"Saved to {filepath}")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    for name, url in URLS.items():
        download_file(name, url)
