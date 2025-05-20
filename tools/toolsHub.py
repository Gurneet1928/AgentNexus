from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_community.utilities import ArxivAPIWrapper
from langchain.agents import tool
import bs4, requests
from langchain_community.utilities.semanticscholar import SemanticScholarAPIWrapper
import yfinance as yf

@tool
def get_word_length(word: str) -> int:
    """
    Returns the length of a word/Count of characters.
    
    Input should be a string. Returns the number of characters in the string.
    
    Examples:
    - "hello" - Returns 5
    - "programming" - Returns 11
    - "AI" - Returns 2
    - "langchain" - Returns 9
    """
    return len(word)

@tool
def python_repl(query: str) -> str:
    """
    Executes Python code and returns the result.
    
    Input should be valid Python code. Returns code output or error message.
    
    Examples:
    - "print(2 + 3)" - Returns "5"
    - "import math; print(math.sqrt(16))" - Returns "4.0"
    - "data = [1, 2, 3]; print(sum(data)/len(data))" - Returns average
    - "for i in range(3): print(f'Item {i}')" - Returns multi-line output
    """
    repl = PythonREPL()
    try:
        result = repl.run(query)
        if not result or result.strip() == "":
            return "Code executed successfully, but produced no output. If you want to see values, use print() statements in your code."
        return result
    except Exception as e:
        return f"Error executing code: {str(e)}"

@tool
def scrapSite(url: str) -> str:
    """
    A Simple Web Scraper that fetches the content of a webpage.
    It uses the requests library to get the HTML content of the page and BeautifulSoup to parse it.
    The function returns the text content of the page, stripping any extra whitespace.
    Input should be a valid URL.

    Examples:
    - "https://en.wikipedia.org/wiki/Python_(programming_language)" - Returns text content of Python's Wikipedia page
    - "https://news.ycombinator.com" - Returns text from Hacker News homepage
    - "https://github.com/blog" - Returns text content from GitHub's blog
    - "https://example.com" - Returns the simple text content from example.com
    """
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = bs4.BeautifulSoup(response.text, 'lxml')
        return soup.body.get_text(' ', strip=True)
    except Exception as e:
        return f"Error scraping site: {str(e)}"
    
@tool
def searchArXiv(query: str) -> str:
    """
    A simple tool to search arXiv for papers related to a given query.
    Input can be a paper name, paper id, author name, or any other relevant keyword.
    If results are found, it returns a string of results.
    For no results, it returns a message indicating that.
    
    Examples:
    - "large language models" - Returns papers about LLMs
    - "1706.03762" - Returns the paper with this arXiv ID (Attention Is All You Need)
    - "Hinton deep learning" - Returns papers by Hinton related to deep learning
    - "quantum computing recent" - Returns recent papers on quantum computing
    """
    arxiv = ArxivAPIWrapper()
    try:
        results = arxiv.run(query)
        if not results:
            return "No results found."
        return results
    except Exception as e:
        return f"Error searching arXiv: {str(e)}"
    
@tool
def sematicScholar(query: str) -> str:
    """
    A simple tool to search Semantic Scholar for papers related to a given query.
    Input can be a paper title, author name, keyword, or research topic.
    If results are found, it returns a string of results.
    For no results, it returns a message indicating that.
    
    Examples:
    - "machine learning" - Returns top papers about machine learning
    - "John Smith neural networks" - Returns papers by John Smith on neural networks
    - "transformer architecture 2023" - Returns recent papers about transformer architectures
    """
    sem_scholar = SemanticScholarAPIWrapper(top_k_results = 3, load_max_docs = 3)
    try:
        results = sem_scholar.run(query)
        if not results:
            return "No results found."
        return results
    except Exception as e:
        return f"Error searching Semantic Scholar: {str(e)}"

@tool
def yahooFinance(input_str: str) -> str:
    """
    A simple tool to fetch stock data from Yahoo Finance.
    
    Input must be in the format: "QUERY | OPERATION"
    
    Where:
    - QUERY: For tickers, use stock symbol (e.g., "AAPL")
             For search, use company name or symbol (e.g., "Apple")
             For market, use market name/location (e.g., "US" or "NYSE")
    
    - OPERATION: Must be one of: 'ticker', 'search', 'market'
    
    Examples:
    - "AAPL | ticker" - Get Apple stock ticker information
    - "Apple | search" - Search for Apple company
    - "NYSE | market" - Get NYSE market information
    """
    parts = input_str.split("|")
    
    if len(parts) != 2:
        return "Invalid input format. Please use: 'QUERY | OPERATION'"
    
    query = parts[0].strip()
    operation = parts[1].strip().lower()
    
    # Rest of the function remains the same
    if operation == 'ticker' or operation == '':
        try:
            ticker = yf.Ticker(query)
            return ticker.info
        except Exception as e:
            return f"Error fetching ticker data: {str(e)}"
    elif operation == 'search':
        try:
            search = yf.Search(query, max_results=3)
            return search.response
        except Exception as e:
            return f"Error Searching for Query: {str(e)}"
    elif operation == 'market':
        try:
            market = yf.Market(query)
            return market.summary
        except Exception as e:
            return f"Error fetching market data: {str(e)}"
    else:
        return "Invalid operation. Please use 'ticker', 'search', or 'market'."