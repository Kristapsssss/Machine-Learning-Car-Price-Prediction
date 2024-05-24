import requests
from bs4 import BeautifulSoup
import pandas as pd

# Car brands above 300 listings as minimum
car_brands = [
    "Audi",
    "BMW",
    "Ford",
    "Honda",
    "Hyundai",
    "Kia",
    "Lexus",
    "Mazda",
    "Mercedes-Benz",
    "Mitsubishi",
    "Nissan",
    "Opel",
    "Peugeot",
    "Renault",
    "Toyota",
    "Volkswagen",
    "Volvo"
]


def get_page_data(url, brand):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    all_rows = soup.select('tr[id^="tr_"]')  # Select rows with id starting with "tr_"

    data = []
    for row in all_rows:
        columns = row.find_all('td')
        if len(columns) > 1:
            model = columns[3].text.strip()
            year = columns[4].text.strip()
            engine_size = columns[5].text.strip()
            mileage = columns[6].text.strip()
            price = columns[7].text.strip()
            posting = {
                'Brand': brand,
                'Model': model,
                'Year': year,
                'Engine Size': engine_size,
                'Mileage': mileage,
                'Price': price
            }
            data.append(posting)

    return data


def get_all_pages_data(url, brand):
    all_data = []
    page = 1
    while True:
        page_url = f"{url}page{page}.html"
        print(f"Scraping page: {page}, the page url is: {page_url}")
        page_data = get_page_data(page_url, brand)

        # Stop if page data is empty or if it's a duplicate of the first page
        if not page_data or (page > 1 and page_data == all_data[:len(page_data)]):
            break

        all_data.extend(page_data)
        page += 1

    return all_data


def scrape_all_brands(base_url, brands):
    all_brands_data = []
    for brand in brands:
        brand_url = f"{base_url}{brand.lower()}/"
        print(f"Scraping brand: {brand}, URL: {brand_url}")
        brand_data = get_all_pages_data(brand_url, brand)
        all_brands_data.extend(brand_data)
    return all_brands_data


base_url = "https://www.ss.lv/lv/transport/cars/"

all_listings = scrape_all_brands(base_url, car_brands)

df = pd.DataFrame(all_listings)
print(df.shape)

df.to_csv('car_listings.csv', index=False)