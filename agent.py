from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from config import OPENAI_API_KEY, OPENAI_MODEL_NAME
from tools.rag_tool import FinancialDocumentSearchTool
from tools.calculator_tool import FinancialCalculatorTool


FINANCIAL_AGENT_SYSTEM_PROMPT = """You are a financial analysis assistant specialized in analyzing SEC 10-K filings.
You have access to a database of 10-K filings from various companies (2010-2019).

Your capabilities:
1. Search and retrieve information from SEC 10-K financial documents
2. Perform financial calculations (growth rates, profit margins, ratios, etc.)

When answering questions:
- Always use the search_financial_documents tool to find relevant information
- If asked to compare or calculate, first retrieve the data, then use the calculator
- Be specific about which company and year the information comes from
- If information is not available, clearly state that

Available companies include: SBUX (Starbucks), MAR (Marriott), and others.
Available years: 2010-2019

Think step by step and use your tools appropriately to provide accurate, well-sourced answers."""


class FinancialAgent:
    """LangChain-based financial analysis agent with RAG and calculator tools."""

    def __init__(self, rag_system, verbose: bool = True):
        """
        Initialize the financial agent.

        Args:
            rag_system: Initialized RAG system for document retrieval
            verbose: Whether to show detailed agent reasoning
        """
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")

        self.verbose = verbose

        # Initialize the LLM
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL_NAME,
            temperature=0,
            api_key=OPENAI_API_KEY,
        )

        # Initialize tools
        self.tools = [
            FinancialDocumentSearchTool(rag_system=rag_system),
            FinancialCalculatorTool(),
        ]

        # Create the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", FINANCIAL_AGENT_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Create the agent
        self.agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt,
        )

        # Create the agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=verbose,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
        )

        # Initialize conversation memory
        self.chat_history: List = []
        
    def _escape_dollar_for_markdown(self, text: str) -> str:
        """Escape $ to avoid Markdown / LaTeX rendering issues."""
        return text.replace("$", "\\$")

    def invoke(self, query: str) -> Dict[str, Any]:
        """
        Process a query and return the agent's response with reasoning steps.

        Args:
            query: The user's question

        Returns:
            Dictionary containing:
                - output: The agent's final response
                - intermediate_steps: List of (tool, tool_input, tool_output) tuples
                - chat_history: Current conversation history
        """
        try:
            result = self.agent_executor.invoke({
                "input": query,
                "chat_history": self.chat_history,
            })

            # Update conversation history
            self.chat_history.append(HumanMessage(content=query))
            self.chat_history.append(AIMessage(content=result["output"]))

            # Format intermediate steps for display
            steps = []
            for step in result.get("intermediate_steps", []):
                action, observation = step
                steps.append({
                    "tool": action.tool,
                    "tool_input": action.tool_input,
                    "observation": observation,
                })

            # 🔧 Escape dollar signs for Markdown rendering
            clean_output = self._escape_dollar_for_markdown(result["output"])

            return {
                "output": clean_output,
                "intermediate_steps": steps,
                "chat_history": self.chat_history,
            }

        except Exception as e:
            error_message = f"Error processing query: {str(e)}"
            return {
                "output": error_message,
                "intermediate_steps": [],
                "chat_history": self.chat_history,
            }

    def clear_history(self):
        """Clear the conversation history."""
        self.chat_history = []

    def get_tools_info(self) -> List[Dict[str, str]]:
        """Get information about available tools."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
            }
            for tool in self.tools
        ]


def create_agent(rag_system, verbose: bool = True) -> FinancialAgent:
    """
    Factory function to create a financial agent.

    Args:
        rag_system: Initialized RAG system
        verbose: Whether to show detailed reasoning

    Returns:
        Configured FinancialAgent instance
    """
    return FinancialAgent(rag_system=rag_system, verbose=verbose)


if __name__ == "__main__":
    # Test the agent
    from system import RAGSystem

    print("Initializing RAG system...")
    rag = RAGSystem()

    print("Creating agent...")
    agent = create_agent(rag, verbose=True)

    print("\nAvailable tools:")
    for tool_info in agent.get_tools_info():
        print(f"  - {tool_info['name']}: {tool_info['description'][:100]}...")

    print("\nTest queries:")
    test_queries = [
        "What was Starbucks' revenue in 2019?",
        "Calculate the growth rate if revenue went from $10 million to $15 million",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        result = agent.invoke(query)
        print(f"Response: {result['output']}")
        if result['intermediate_steps']:
            print("Tools used:")
            for step in result['intermediate_steps']:
                print(f"  - {step['tool']}: {step['tool_input']}")
