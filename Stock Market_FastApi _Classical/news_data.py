from newsapi import NewsApiClient
import yfinance as yf
from datetime import datetime, timedelta

# Define the maximum allowed date for your News API plan
MAX_ALLOWED_DATE = datetime(2024, 11, 28)

def get_stock_news_titles(date: datetime, config: dict):
    # Adjust the date if it exceeds the maximum allowed date
    if date > MAX_ALLOWED_DATE:
        date = MAX_ALLOWED_DATE

    # Initialize the NewsApiClient with your API key
    newsapi = NewsApiClient(api_key=config['news_api_key'])

    stock = yf.Ticker(config['stock_symbol'])
    stock_info = stock.info
    stock_name = stock_info.get('longName', config['stock_symbol'])

    # Adjust the date to avoid fetching news before the max allowed date
    previous_date = max(date - timedelta(days=1), MAX_ALLOWED_DATE)
    start_date = previous_date.strftime("%Y-%m-%d")
    end_date = date.strftime("%Y-%m-%d")

    # Fetch articles from the News API
    all_articles = newsapi.get_everything(
        q=stock_name,
        from_param=start_date,
        to=end_date,
        language='en',
        sort_by='relevancy'
    )

    # Extract the titles from the articles
    titles = [article['title'] for article in all_articles['articles']]
    return titles