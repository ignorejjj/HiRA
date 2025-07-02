import os
import json
import requests
from requests.exceptions import Timeout
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import concurrent
from concurrent.futures import ThreadPoolExecutor
import pdfplumber
from io import BytesIO
import re
import string
from typing import Optional, Tuple
from nltk.tokenize import sent_tokenize
from typing import List, Dict, Union
from urllib.parse import urljoin
import aiohttp
import asyncio
import chardet
import langid
import random
from urllib.parse import urlencode

WebParserClient_url = None

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0',
    'Referer': 'https://www.google.com/',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',}
# ----------------------- Custom Headers -----------------------
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/58.0.3029.110 Safari/537.36',
    'Referer': 'https://www.google.com/',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}

# Initialize session
session = requests.Session()
session.headers.update(headers)

error_indicators = [
    'limit exceeded',
    'Error fetching',
    'Account balance not enough',
    'Invalid bearer token',
    'HTTP error occurred',
    'Error: Connection error occurred',
    'Error: Request timed out',
    'Unexpected error',
    'Please turn on Javascript',
    'Enable JavaScript',
    'port=443',
    'Please enable cookies',
]

class WebParserClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        
    def parse_urls(self, urls: List[str], timeout: int = 120) -> List[Dict[str, Union[str, bool]]]:
        endpoint = urljoin(self.base_url, "/parse_urls")
        response = requests.post(endpoint, json={"urls": urls}, timeout=timeout)
        response.raise_for_status()  
        
        return response.json()["results"]


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))

def f1_score(true_set: set, pred_set: set) -> float:
    intersection = len(true_set.intersection(pred_set))
    if not intersection:
        return 0.0
    precision = intersection / float(len(pred_set))
    recall = intersection / float(len(true_set))
    return 2 * (precision * recall) / (precision + recall)

def extract_snippet_with_context(full_text: str, snippet: str, context_chars: int = 3000) -> Tuple[bool, str]:
    try:
        full_text = full_text[:100000]

        snippet = snippet.lower()
        snippet = remove_punctuation(snippet)
        snippet_words = set(snippet.split())

        best_sentence = None
        best_f1 = 0.2

        # sentences = re.split(r'(?<=[.!?]) +', full_text)  # Split sentences using regex, supporting ., !, ? endings
        sentences = sent_tokenize(full_text)  # Split sentences using nltk's sent_tokenize

        for sentence in sentences:
            key_sentence = sentence.lower()
            key_sentence = remove_punctuation(key_sentence)
            sentence_words = set(key_sentence.split())
            f1 = f1_score(snippet_words, sentence_words)
            if f1 > best_f1:
                best_f1 = f1
                best_sentence = sentence

        if best_sentence:
            para_start = full_text.find(best_sentence)
            para_end = para_start + len(best_sentence)
            start_index = max(0, para_start - context_chars)
            end_index = min(len(full_text), para_end + context_chars)
            # if end_index - start_index < 2 * context_chars:
            #     end_index = min(len(full_text), start_index + 2 * context_chars)
            context = full_text[start_index:end_index]
            return True, context
        else:
            # If no matching sentence is found, return the first context_chars*2 characters of the full text
            return False, full_text[:context_chars * 2]
    except Exception as e:
        return False, f"Failed to extract snippet context due to {str(e)}"

def extract_text_from_url(url, use_jina=False, jina_api_key=None, snippet: Optional[str] = None, keep_links=False):
    if url.startwith('https://arxiv.org/abs/'):
        url = url.replace('https://arxiv.org/abs/', 'https://ar5iv.labs.arxiv.org/html/')
    try:
        if use_jina:
            jina_headers = {
                'Authorization': f'Bearer {jina_api_key}',
                'X-Return-Format': 'markdown',
            }
            response = requests.get(f'https://r.jina.ai/{url}', headers=jina_headers).text
            # Remove URLs
            pattern = r"\(https?:.*?\)|\[https?:.*?\]"
            text = re.sub(pattern, "", response).replace('---','-').replace('===','=').replace('   ',' ').replace('   ',' ')
        else:
            if 'pdf' in url:
                return extract_pdf_text(url)

            try:
                response = session.get(url, timeout=30)
                response.raise_for_status()
                
                if response.encoding.lower() == 'iso-8859-1':
                    response.encoding = response.apparent_encoding
                
                try:
                    soup = BeautifulSoup(response.text, 'lxml')
                except Exception:
                    soup = BeautifulSoup(response.text, 'html.parser')

                has_error = (any(indicator.lower() in response.text.lower() for indicator in error_indicators) and len(response.text.split()) < 64) or response.text == ''
                if has_error and WebParserClient_url is not None:
                    client = WebParserClient(WebParserClient_url)
                    results = client.parse_urls([url])
                    if results and results[0]["success"]:
                        text = results[0]["content"]
                    else:
                        error_msg = results[0].get("error", "Unknown error") if results else "No results returned"
                        return f"WebParserClient error: {error_msg}"
                else:
                    if keep_links:
                        for element in soup.find_all(['script', 'style', 'meta', 'link']):
                            element.decompose()

                        text_parts = []
                        for element in soup.body.descendants if soup.body else soup.descendants:
                            if isinstance(element, str) and element.strip():
                                cleaned_text = ' '.join(element.strip().split())
                                if cleaned_text:
                                    text_parts.append(cleaned_text)
                            elif element.name == 'a' and element.get('href'):
                                href = element.get('href')
                                link_text = element.get_text(strip=True)
                                if href and link_text:  
                                    if href.startswith('/'):
                                        base_url = '/'.join(url.split('/')[:3])
                                        href = base_url + href
                                    elif not href.startswith(('http://', 'https://')):
                                        href = url.rstrip('/') + '/' + href
                                    text_parts.append(f"[{link_text}]({href})")

                        text = ' '.join(text_parts)
                        text = ' '.join(text.split())
                    else:
                        text = soup.get_text(separator=' ', strip=True)
            except Exception as e:
                if WebParserClient_url is None:
                    return f"Error extracting content: {str(e)}"
                client = WebParserClient(WebParserClient_url)
                results = client.parse_urls([url])
                if results and results[0]["success"]:
                    text = results[0]["content"]
                else:
                    error_msg = results[0].get("error", "Unknown error") if results else "No results returned"
                    return f"WebParserClient error: {error_msg}"

        if snippet:
            success, context = extract_snippet_with_context(text, snippet)
            if success:
                return context
            else:
                return text
        else:
            return text[:20000]
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except requests.exceptions.ConnectionError:
        return "Error: Connection error occurred"
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def fetch_page_content(urls, max_workers=32, use_jina=False, jina_api_key=None, snippets: Optional[dict] = None, show_progress=False, keep_links=False):
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_text_from_url, url, use_jina, jina_api_key, snippets.get(url) if snippets else None, keep_links): url
            for url in urls
        }
        completed_futures = concurrent.futures.as_completed(futures)
        if show_progress:
            completed_futures = tqdm(completed_futures, desc="Fetching URLs", total=len(urls))
            
        for future in completed_futures:
            url = futures[future]
            try:
                data = future.result()
                results[url] = data
            except Exception as exc:
                results[url] = f"Error fetching {url}: {exc}"
            # time.sleep(0.1)  # Simple rate limiting
    return results

def bing_web_search(query, subscription_key, endpoint, market='en-US', language='en', timeout=20):
    """
    Perform a search using the Bing Web Search API with a set timeout.

    Args:
        query (str): Search query.
        subscription_key (str): Subscription key for the Bing Search API.
        endpoint (str): Endpoint for the Bing Search API.
        market (str): Market, e.g., "en-US" or "zh-CN".
        language (str): Language of the results, e.g., "en".
        timeout (int or float or tuple): Request timeout in seconds.
                                         Can be a float representing the total timeout,
                                         or a tuple (connect timeout, read timeout).

    Returns:
        dict: JSON response of the search results. Returns empty dict if all retries fail.
    """
    headers = {
        "Ocp-Apim-Subscription-Key": subscription_key
    }
    params = {
        "q": query,
        "mkt": market,
        "setLang": language,
        "textDecorations": True,
        "textFormat": "HTML"
    }

    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            response = requests.get(endpoint, headers=headers, params=params, timeout=timeout)
            response.raise_for_status()  # Raise exception if the request failed
            search_results = response.json()
            return search_results
        except Timeout:
            retry_count += 1
            if retry_count == max_retries:
                print(f"Bing Web Search request timed out ({timeout} seconds) for query: {query} after {max_retries} retries")
                return {}
            print(f"Bing Web Search Timeout occurred, retrying ({retry_count}/{max_retries})...")
        except requests.exceptions.RequestException as e:
            retry_count += 1
            if retry_count == max_retries:
                print(f"Bing Web Search Request Error occurred: {e} after {max_retries} retries")
                return {}
            print(f"Bing Web Search Request Error occurred, retrying ({retry_count}/{max_retries})...")
        time.sleep(1)  # Wait 1 second between retries
    
    return {}  # Should never reach here but added for completeness

def extract_pdf_text(url):
    """
    Extract text from a PDF.

    Args:
        url (str): URL of the PDF file.

    Returns:
        str: Extracted text content or error message.
    """
    try:
        response = session.get(url, timeout=20)  # Set timeout to 20 seconds
        if response.status_code != 200:
            return f"Error: Unable to retrieve the PDF (status code {response.status_code})"
        
        # Open the PDF file using pdfplumber
        with pdfplumber.open(BytesIO(response.content)) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text
        
        # Limit the text length
        cleaned_text = full_text
        return cleaned_text
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Error: {str(e)}"




def extract_relevant_info(search_results):
    """
    Extract relevant information from Bing search results.

    Args:
        search_results (dict): JSON response from the Bing Web Search API.

    Returns:
        list: A list of dictionaries containing the extracted information.
    """
    useful_info = []
    
    if 'webPages' in search_results and 'value' in search_results['webPages']:
        for id, result in enumerate(search_results['webPages']['value']):
            info = {
                'id': id + 1,  # Increment id for easier subsequent operations
                'title': result.get('name', ''),
                'url': result.get('url', ''),
                'site_name': result.get('siteName', ''),
                'date': result.get('datePublished', '').split('T')[0],
                'snippet': result.get('snippet', ''),  # Remove HTML tags
                # Add context content to the information
                'context': ''  # Reserved field to be filled later
            }
            useful_info.append(info)
    # useful_info = []
    # if 'organic' not in search_results:
    #     if 'webPages' in search_results and 'value' in search_results['webPages']:
    #         for id, result in enumerate(search_results['webPages']['value']):
    #             info = {
    #                 'id': id + 1,  # Increment id for easier subsequent operations
    #                 'title': result.get('name', ''),
    #                 'url': result.get('url', ''),
    #                 'site_name': result.get('siteName', ''),
    #                 'date': result.get('datePublished', '').split('T')[0],
    #                 'snippet': result.get('snippet', ''),  # Remove HTML tags
    #                 # Add context content to the information
    #                 'context': ''  # Reserved field to be filled later
    #             }
    #             useful_info.append(info)
    #     if useful_info == []:
    #         print("Have error in bing_web_search: no organic results (search_results = {})".format(search_results))
    #         return []
    #     else:
    #         return useful_info
    # for id, item in enumerate(search_results['organic']):
    #     info = {
    #         'id': id + 1,  # Increment id for easier subsequent operations
    #         'title': item.get('title', ''),
    #         'url': item.get('link', ''),
    #         'site_name': item.get('site_name', ''),
    #         'snippet': item.get('description', ''),  # Remove HTML tags
    #         'context': ''  # Reserved field to be filled later
    #     }
    #     useful_info.append(info)
    return useful_info


async def bing_web_search_async(query, subscription_key, endpoint, market='en-US', language='en', timeout=20):
    """
    Perform an asynchronous search using the Bing Web Search API.

    Args:
        query (str): Search query.
        subscription_key (str): Subscription key for the Bing Search API.
        endpoint (str): Endpoint for the Bing Search API.
        market (str): Market, e.g., "en-US" or "zh-CN".
        language (str): Language of the results, e.g., "en".
        timeout (int): Request timeout in seconds.

    Returns:
        dict: JSON response of the search results. Returns empty dict if all retries fail.
    """
    headers = {
        "Ocp-Apim-Subscription-Key": subscription_key
    }
    params = {
        "q": query.strip("\""),
        "mkt": market,
        "setLang": language,
        "textDecorations": True,
        "textFormat": "HTML"
    }

    max_retries = 5
    retry_count = 0

    while retry_count < max_retries:
        try:
            response = session.get(endpoint, headers=headers, params=params, timeout=timeout)
            response.raise_for_status()
            search_results = response.json()
            return search_results
        except Exception as e:
            retry_count += 1
            if retry_count == max_retries:
                print(f"Bing Web Search Request Error occurred: {e} after {max_retries} retries")
                return {}
            print(f"Bing Web Search Request Error occurred, retrying ({retry_count}/{max_retries})...")
            time.sleep(1)  # Wait 1 second between retries

    return {}

class RateLimiter:
    def __init__(self, rate_limit: int, time_window: int = 60):
        self.rate_limit = rate_limit
        self.time_window = time_window
        self.tokens = rate_limit
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            while self.tokens <= 0:
                now = time.time()
                time_passed = now - self.last_update
                self.tokens = min(
                    self.rate_limit,
                    self.tokens + (time_passed * self.rate_limit / self.time_window)
                )
                self.last_update = now
                if self.tokens <= 0:
                    await asyncio.sleep(random.randint(5, 30))  # 等待xxx秒后重试
            
            self.tokens -= 1
            return True

jina_rate_limiter = RateLimiter(rate_limit=130)  # 每分钟xxx次，避免报错

async def extract_text_from_url_async(url: str, session: aiohttp.ClientSession, use_jina: bool = False, 
                                    jina_api_key: Optional[str] = None, snippet: Optional[str] = None, 
                                    keep_links: bool = False) -> str:
    """Async version of extract_text_from_url"""
    try:
        if use_jina:
            await jina_rate_limiter.acquire()
            
            jina_headers = {
                'Authorization': f'Bearer {jina_api_key}',
                'X-Return-Format': 'markdown',
            }
            async with session.get(f'https://r.jina.ai/{url}', headers=jina_headers) as response:
                text = await response.text()
                if not keep_links:
                    pattern = r"\(https?:.*?\)|\[https?:.*?\]"
                    text = re.sub(pattern, "", text)
                text = text.replace('---','-').replace('===','=').replace('   ',' ').replace('   ',' ')
        else:
            if 'pdf' in url:
                text = await extract_pdf_text_async(url, session)
                return text[:10000]

            async with session.get(url) as response:
                content_type = response.headers.get('content-type', '').lower()
                if 'charset' in content_type:
                    charset = content_type.split('charset=')[-1]
                    html = await response.text(encoding=charset)
                else:
                    content = await response.read()
                    detected = chardet.detect(content)
                    encoding = detected['encoding'] if detected['encoding'] else 'utf-8'
                    html = content.decode(encoding, errors='replace')
                
                has_error = (any(indicator.lower() in html.lower() for indicator in error_indicators) and len(html.split()) < 64) or len(html) < 50 or len(html.split()) < 20
                if has_error and WebParserClient_url is not None:
                    client = WebParserClient(WebParserClient_url)
                    results = client.parse_urls([url])
                    if results and results[0]["success"]:
                        text = results[0]["content"]
                    else:
                        error_msg = results[0].get("error", "Unknown error") if results else "No results returned"
                        return f"WebParserClient error: {error_msg}"
                else:
                    try:
                        soup = BeautifulSoup(html, 'lxml')
                    except Exception:
                        soup = BeautifulSoup(html, 'html.parser')

                    if keep_links:
                        for element in soup.find_all(['script', 'style', 'meta', 'link']):
                            element.decompose()

                        text_parts = []
                        for element in soup.body.descendants if soup.body else soup.descendants:
                            if isinstance(element, str) and element.strip():
                                cleaned_text = ' '.join(element.strip().split())
                                if cleaned_text:
                                    text_parts.append(cleaned_text)
                            elif element.name == 'a' and element.get('href'):
                                href = element.get('href')
                                link_text = element.get_text(strip=True)
                                if href and link_text:
                                    if href.startswith('/'):
                                        base_url = '/'.join(url.split('/')[:3])
                                        href = base_url + href
                                    elif not href.startswith(('http://', 'https://')):
                                        href = url.rstrip('/') + '/' + href
                                    text_parts.append(f"[{link_text}]({href})")

                        text = ' '.join(text_parts)
                        text = ' '.join(text.split())
                    else:
                        text = soup.get_text(separator=' ', strip=True)

        if snippet:
            success, context = extract_snippet_with_context(text, snippet)
            return context if success else text
        else:
            return text[:50000]

    except Exception as e:
        return f"Error fetching {url}: {str(e)}"

async def fetch_page_content_async(urls: List[str], use_jina: bool = False, jina_api_key: Optional[str] = None, 
                                 snippets: Optional[Dict[str, str]] = None, show_progress: bool = False,
                                 keep_links: bool = False, max_concurrent: int = 32) -> Dict[str, str]:
    """Asynchronously fetch content from multiple URLs."""
    async def process_urls():
        connector = aiohttp.TCPConnector(limit=max_concurrent)
        timeout = aiohttp.ClientTimeout(total=240)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout, headers=HEADERS) as session:
            tasks = []
            for url in urls:
                task = extract_text_from_url_async(
                    url, 
                    session, 
                    use_jina, 
                    jina_api_key,
                    snippets.get(url) if snippets else None,
                    keep_links
                )
                tasks.append(task)
            
            if show_progress:
                results = []
                for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Fetching URLs"):
                    result = await task
                    results.append(result)
            else:
                results = await asyncio.gather(*tasks)
            
            return {url: result for url, result in zip(urls, results)} 

    return await process_urls() 

async def extract_pdf_text_async(url: str, session: aiohttp.ClientSession) -> str:
    """
    Asynchronously extract text from a PDF.

    Args:
        url (str): URL of the PDF file.
        session (aiohttp.ClientSession): Aiohttp client session.

    Returns:
        str: Extracted text content or error message.
    """
    try:
        async with session.get(url, timeout=30) as response:  # Set timeout to 20 seconds
            if response.status != 200:
                return f"Error: Unable to retrieve the PDF (status code {response.status})"
            
            content = await response.read()
            
            with pdfplumber.open(BytesIO(content)) as pdf:
                full_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text
            
            cleaned_text = full_text
            return cleaned_text
    except asyncio.TimeoutError:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Error: {str(e)}"




# ------------------------------------------------------------

if __name__ == "__main__":
    query = "who is the president of the united states"
    
    BING_SUBSCRIPTION_KEY = ""
    if not BING_SUBSCRIPTION_KEY:
        raise ValueError("Please set the BING_SEARCH_V7_SUBSCRIPTION_KEY environment variable.")
    
    bing_endpoint = "https://api.bing.microsoft.com/v7.0/search"
    
    # Perform the search
    print("Performing Bing Web Search...")
    search_results = bing_web_search(query, BING_SUBSCRIPTION_KEY, bing_endpoint)
    print(search_results)
    
    # print("Extracting relevant information from search results...")
    # extracted_info = extract_relevant_info(search_results)

    # print("Fetching and extracting context for each snippet...")
    # for info in tqdm(extracted_info, desc="Processing Snippets"):
    #     full_text = extract_text_from_url(info['url'], use_jina=True)  # Get full webpage text
    #     if full_text and not full_text.startswith("Error"):
    #         success, context = extract_snippet_with_context(full_text, info['snippet'])
    #         if success:
    #             info['context'] = context
    #         else:
    #             info['context'] = f"Could not extract context. Returning first 8000 chars: {full_text[:8000]}"
    #     else:
    #         info['context'] = f"Failed to fetch full text: {full_text}"

    # print("Your Search Query:", query)
    # print("Final extracted information with context:")
    # print(json.dumps(extracted_info, indent=2, ensure_ascii=False))
