from mcp.server.fastmcp import FastMCP
import logging
import sys
import os

from mcp_server.logic.client import searx_client
from mcp_server.logic.processor import ResponseProcessor

LOG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../orchestrator.log"))
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'), 
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("MCP-Orchestrator")

mcp = FastMCP("SearXNG-Local-Search-Tool")

@mcp.tool()
async def search_smart(
    query: str, 
    limit: int = 5,
    language: str = "ru",
    categories: str = "general"
) -> str:
    """
    Tool for searching information via the local SearxNG meta-search engine.
    
    :param query: The search query (e.g., "how does MCP protocol work")
    :param limit: Number of results to return (default is 5)
    :param language: Language of the results (e.g., 'ru', 'en', 'all')
    :param categories: Search categories ('general', 'it', 'science', 'news')
    """
    logger.info(f"🔎 LLM requested search: '{query}' | Lang: {language} | Categories: {categories}")
    
    try:
        logger.info("Delegating request to SearxLocalClient...")
        raw_data = await searx_client.search(query, language, categories)
        
        logger.info("Passing raw data to ResponseProcessor...")
        results = raw_data.get("results", [])
        clean_text = ResponseProcessor.process_results(results, limit=limit)
        
        logger.info("Successfully processed and returned results to LLM.")
        return clean_text

    except Exception as e:
        error_msg = (
            f"System Search Error: {str(e)}\n"
            "Check if the Docker container is running via `docker-compose up -d` "
            "and verify accessibility at http://127.0.0.1:8080."
        )
        logger.error(f"Failed to execute search tool: {repr(e)}")
        return error_msg

if __name__ == "__main__":
    logger.info("Starting Local SearXNG MCP Server...")
    mcp.run()