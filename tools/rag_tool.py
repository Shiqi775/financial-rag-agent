from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

class FinancialDocumentSearchInput(BaseModel):
    """Input schema for financial document search tool."""
    query: str = Field(description="The search query to find relevant information in SEC 10-K filings")
    ticker: Optional[str] = Field(default=None, description="Company ticker symbol (e.g., SBUX, AAPL). If not provided, searches all companies.")
    year: Optional[str] = Field(default=None, description="Year of the filing (2010-2019). If not provided, searches all years.")


class FinancialDocumentSearchTool(BaseTool):
    """Tool for searching SEC 10-K financial documents using RAG."""

    name: str = "search_financial_documents"
    description: str = """Search SEC 10-K financial filings for company information.
    Use this tool to find information about:
    - Revenue, profit, and financial metrics
    - Business operations and strategy
    - Risk factors and challenges
    - Management discussion and analysis
    - Any other information from annual reports

    You can optionally filter by company ticker (e.g., SBUX, MAR, AAPL) and year (2010-2019).
    If ticker or year is not specified, the search will cover all available documents."""

    args_schema: Type[BaseModel] = FinancialDocumentSearchInput
    rag_system: object = None  # Will be set during initialization

    def __init__(self, rag_system, **kwargs):
        super().__init__(**kwargs)
        self.rag_system = rag_system

    def _run(self, query: str, ticker: Optional[str] = None, year: Optional[str] = None) -> str:
        """Execute the document search."""
        try:
            # If ticker and year are provided, use filtered search
            if ticker and year:
                response, nodes = self.rag_system.retrieve_and_respond(query, ticker, year)
                if nodes:
                    source_info = f"\n\nSource: {ticker} {year} 10-K filing"
                    return response + source_info
                return response

            # If only ticker is provided, search across all years
            elif ticker:
                results = []
                available_years = [str(y) for y in range(2010, 2020)]
                for yr in available_years:
                    try:
                        response, nodes = self.rag_system.retrieve_and_respond(query, ticker, yr)
                        if nodes and "No relevant information" not in response:
                            results.append(f"[{yr}]: {response}")
                    except Exception:
                        continue
                if results:
                    return f"Results for {ticker}:\n" + "\n".join(results[:3])  # Limit to top 3 results
                return f"No relevant information found for {ticker}."

            # If only year is provided, search across all tickers
            elif year:
                results = []
                available_tickers = self.rag_system.get_available_tickers()
                for tk in available_tickers[:5]:  # Limit to first 5 tickers
                    try:
                        response, nodes = self.rag_system.retrieve_and_respond(query, tk, year)
                        if nodes and "No relevant information" not in response:
                            results.append(f"[{tk}]: {response}")
                    except Exception:
                        continue
                if results:
                    return f"Results for {year}:\n" + "\n".join(results[:3])
                return f"No relevant information found for year {year}."

            # No filters provided - use a default search
            else:
                return "Please specify at least a ticker symbol or year to search financial documents."

        except Exception as e:
            return f"Error searching documents: {str(e)}"

    async def _arun(self, query: str, ticker: Optional[str] = None, year: Optional[str] = None) -> str:
        """Async version of the tool."""
        return self._run(query, ticker, year)
