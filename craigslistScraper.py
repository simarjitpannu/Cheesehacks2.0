import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_craigslist_free_items():
    # URL for the 'free' section in Madison, WI
    url = "https://madison.craigslist.org/d/free-stuff/search/zip"

    # Send a GET request to fetch the page content
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Failed to fetch the page: {response.status_code}")
        return

    # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all the listings
    listings = soup.find_all('li', class_='cl-static-search-result')
    print(f"Number of listings found: {len(listings)}")  # Debugging log

    # Extract titles and links
    items = []
    for listing in listings:
        try:
            title_element = listing.find('div', class_='title')
            title = title_element.get_text(strip=True) if title_element else "N/A"

            link_element = listing.find('a', href=True)
            link = link_element['href'] if link_element else "N/A"

            items.append({"Title": title, "Link": link})
        except Exception as e:
            print(f"Error parsing listing: {e}")

    # Save the data to a CSV file
    if items:
        df = pd.DataFrame(items)
        df.to_csv('craigslist_free_items.csv', index=False)
        print("Data saved to craigslist_free_items.csv")
    else:
        print("No items were found. Check the HTML structure and selectors.")

# Run the scraper
scrape_craigslist_free_items()
