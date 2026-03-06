import httpx
import logging
import asyncio
import time

logger = logging.getLogger("SearxLocalClient")

class SearxClient:
    """
    HTTP Client to interact with the local SearxNG Docker instance.
    Includes retry logic and timeout handling.
    """
    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        self.base_url = base_url
        self.timeout = 4.0

    async def search(self, query: str, language: str, categories: str) -> dict:
        """
        Executes a search query against the local SearxNG instance.
        """
        params = {
            "q": query,
            "format": "json",
            "language": language,
            "categories": categories
        }
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting local search (Attempt {attempt + 1}/{max_retries}) | Query: '{query}'")
                start_time = time.perf_counter()
                
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.get(f"{self.base_url}/search", params=params)
                    
                    if response.status_code != 200:
                        raise Exception(f"HTTP {response.status_code}: {response.text}")
                    
                    data = response.json()
                    duration = time.perf_counter() - start_time
                    results_count = len(data.get('results', []))
                    
                    logger.info(f"✅ Local search successful in {duration:.2f}s | Results found: {results_count}")
                    return data
                    
            except httpx.ConnectError:
                logger.error("❌ Connection Error! Ensure the Docker container is running at http://127.0.0.1:8080.")
                if attempt == max_retries - 1:
                    raise Exception("Unable to connect to the local SearxNG container.")
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.warning(f"⚠️ Search error (Attempt {attempt+1}): {repr(e)}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(1)

searx_client = SearxClient()