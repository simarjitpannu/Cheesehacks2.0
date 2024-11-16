import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_craigslist_free_items(location):
    # Create the Craigslist URL dynamically based on the location
    url = f"https://{location}.craigslist.org/d/free-stuff/search/zip"

    # Send a GET request to fetch the page content
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Failed to fetch the page for location '{location}': {response.status_code}")
        return

    # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all the listings
    listings = soup.find_all('li', class_='cl-static-search-result')
    print(f"Number of listings found for {location}: {len(listings)}")  # Debugging log

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
        output_file = f'craigslist_free_items_{location}.csv'
        df = pd.DataFrame(items)
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
    else:
        print(f"No items were found for location '{location}'. Check the HTML structure and selectors.")

# Example usage
location = input("Enter the Craigslist location (e.g., madison, chicago, newyork): ").strip()
scrape_craigslist_free_items(location)
