"""
Custom tools for the LangChain agent.
Includes Tavily search, PubMed search, Wikipedia search, and RAG retriever.
"""

import requests
import xmltodict
from typing import Any, Optional

from langchain_core.tools import tool

from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Module-level config reference (set by create_tools) ────────────
_config: Optional[Config] = None


def _set_config(config: Config) -> None:
    global _config
    _config = config


# ── PubMed Search ──────────────────────────────────────────────────

@tool
def search_pubmed(query: str) -> str:
    """
    Search PubMed for medical research articles related to a skin condition.
    Returns titles and summaries of the top 3 results.
    Use this tool when you need evidence-based medical research data.

    Args:
        query: Medical search query (e.g. "melanoma treatment options").
    """
    try:
        # Step 1: Search for article IDs
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": 3,
            "sort": "relevance",
            "retmode": "json",
        }
        search_resp = requests.get(search_url, params=search_params, timeout=10)
        search_data = search_resp.json()

        id_list = search_data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return "No PubMed articles found for this query."

        # Step 2: Fetch article summaries
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "rettype": "abstract",
            "retmode": "xml",
        }
        fetch_resp = requests.get(fetch_url, params=fetch_params, timeout=10)
        articles = xmltodict.parse(fetch_resp.content)

        results = []
        article_set = articles.get("PubmedArticleSet", {}).get("PubmedArticle", [])
        if isinstance(article_set, dict):
            article_set = [article_set]

        for i, article in enumerate(article_set[:3], 1):
            medline = article.get("MedlineCitation", {})
            article_data = medline.get("Article", {})
            title = article_data.get("ArticleTitle", "No title")
            if isinstance(title, dict):
                title = title.get("#text", "No title")

            abstract_data = article_data.get("Abstract", {}).get("AbstractText", "No abstract available")
            if isinstance(abstract_data, list):
                abstract_parts = []
                for part in abstract_data:
                    if isinstance(part, dict):
                        abstract_parts.append(part.get("#text", ""))
                    else:
                        abstract_parts.append(str(part))
                abstract = " ".join(abstract_parts)
            elif isinstance(abstract_data, dict):
                abstract = abstract_data.get("#text", "No abstract available")
            else:
                abstract = str(abstract_data)

            pmid = medline.get("PMID", {})
            if isinstance(pmid, dict):
                pmid = pmid.get("#text", "")

            results.append(
                f"{i}. **{title}**\n"
                f"   PMID: {pmid}\n"
                f"   {abstract[:400]}..."
            )

        return "\n\n".join(results)

    except Exception as e:
        logger.error(f"PubMed search failed: {e}")
        return f"PubMed search failed: {str(e)}"


# ── Wikipedia Medical Search ───────────────────────────────────────

@tool
def search_wikipedia(query: str) -> str:
    """
    Search Wikipedia for medical information about a skin condition.
    Returns a summary of the most relevant article.
    Use this for general medical context and definitions.

    Args:
        query: Search query (e.g. "basal cell carcinoma").
    """
    try:
        import wikipedia

        wikipedia.set_lang("en")
        results = wikipedia.search(query + " skin dermatology", results=3)

        if not results:
            return "No relevant Wikipedia articles found."

        summaries = []
        for title in results[:2]:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                summary = page.summary[:500]
                summaries.append(f"**{page.title}**\n{summary}\nSource: {page.url}")
            except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
                continue

        if not summaries:
            return "Could not retrieve Wikipedia articles for this query."

        return "\n\n---\n\n".join(summaries)

    except Exception as e:
        logger.error(f"Wikipedia search failed: {e}")
        return f"Wikipedia search failed: {str(e)}"


# ── RAG Retriever Tool ─────────────────────────────────────────────

@tool
def search_medical_knowledge(query: str) -> str:
    """
    Search the internal trusted medical knowledge base for information about skin conditions.
    This searches curated content from WHO, NIH, Mayo Clinic, and other trusted sources.
    Use this as the PRIMARY source of medical information.

    Args:
        query: Medical query (e.g. "melanoma risk factors and treatment").
    """
    try:
        if _config is None:
            return "Medical knowledge base not initialized. Please configure API keys."

        from rag.vector_store import get_retriever

        retriever = get_retriever(_config)
        docs = retriever.invoke(query)

        if not docs:
            return "No relevant medical knowledge found in the knowledge base."

        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            topic = doc.metadata.get("topic", "Unknown")
            results.append(
                f"{i}. [{source}] **{topic}**\n{doc.page_content}"
            )

        return "\n\n---\n\n".join(results)

    except Exception as e:
        logger.error(f"Medical knowledge search failed: {e}")
        return f"Medical knowledge search failed: {str(e)}"


# ── Tavily Web Search Tool ─────────────────────────────────────────

@tool
def search_web_medical(query: str) -> str:
    """
    Search the web for recent medical information about skin conditions using Tavily.
    Use this for recent news, guidelines, or information not in the knowledge base.

    Args:
        query: Medical web search query.
    """
    try:
        if _config is None or not _config.has_tavily:
            return "Tavily API key not configured. Web search is unavailable."

        from tavily import TavilyClient

        client = TavilyClient(api_key=_config.tavily_api_key)

        response = client.search(
            query=query + " dermatology skin disease",
            search_depth="basic",
            max_results=3,
            include_answer=True,
        )

        results = []
        if response.get("answer"):
            results.append(f"**Summary:** {response['answer']}")

        for item in response.get("results", [])[:3]:
            title = item.get("title", "No title")
            content = item.get("content", "")[:300]
            url = item.get("url", "")
            results.append(f"**{title}**\n{content}\nSource: {url}")

        return "\n\n---\n\n".join(results) if results else "No web results found."

    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        return f"Web search failed: {str(e)}"


# ── Tool Factory ───────────────────────────────────────────────────

def create_tools(config: Config) -> list:
    """
    Create and return all agent tools.

    Args:
        config: Application configuration.

    Returns:
        List of LangChain tool instances.
    """
    _set_config(config)

    tools = [
        search_medical_knowledge,
        search_pubmed,
        search_wikipedia,
    ]

    if config.has_tavily:
        tools.append(search_web_medical)
    else:
        logger.warning("Tavily API key not set — web search tool will be unavailable")

    logger.info(f"Created {len(tools)} agent tools")
    return tools
