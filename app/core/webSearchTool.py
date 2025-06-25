from langchain_tavily import TavilySearch

tool = TavilySearch(max_results=5)
tools = [tool]