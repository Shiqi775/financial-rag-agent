from typing import Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
import re


class FinancialCalculatorInput(BaseModel):
    """Input schema for financial calculator tool."""
    expression: str = Field(
        description="""The financial calculation to perform. Supported operations:
        - Basic math: addition (+), subtraction (-), multiplication (*), division (/)
        - Growth rate: 'growth_rate(old_value, new_value)' - calculates percentage change
        - Profit margin: 'profit_margin(profit, revenue)' - calculates profit as percentage of revenue
        - Percentage: 'percentage(part, whole)' - calculates what percentage part is of whole
        - Ratio: 'ratio(numerator, denominator)' - calculates simple ratio

        Examples:
        - '1000000 + 500000'
        - 'growth_rate(10000000, 15000000)'
        - 'profit_margin(2000000, 10000000)'
        - 'percentage(25000, 100000)'
        - 'ratio(5000000, 10000000)'
        """
    )


class FinancialCalculatorTool(BaseTool):
    """Tool for performing financial calculations."""

    name: str = "financial_calculator"
    description: str = """Perform financial calculations including:
    - Basic arithmetic (addition, subtraction, multiplication, division)
    - Growth rate calculations: growth_rate(old_value, new_value)
    - Profit margin: profit_margin(profit, revenue)
    - Percentage calculations: percentage(part, whole)
    - Ratio calculations: ratio(numerator, denominator)

    Use this tool when you need to compute financial metrics, compare values, or perform math operations on numbers extracted from documents."""

    args_schema: Type[BaseModel] = FinancialCalculatorInput

    def _run(self, expression: str) -> str:
        """Execute the financial calculation."""
        try:
            # Clean the expression
            expression = expression.strip()

            # Handle financial functions
            if "growth_rate" in expression.lower():
                return self._calculate_growth_rate(expression)
            elif "profit_margin" in expression.lower():
                return self._calculate_profit_margin(expression)
            elif "percentage" in expression.lower():
                return self._calculate_percentage(expression)
            elif "ratio" in expression.lower():
                return self._calculate_ratio(expression)
            else:
                # Basic arithmetic - safely evaluate
                return self._safe_eval(expression)

        except Exception as e:
            return f"Calculation error: {str(e)}. Please check your expression format."

    def _extract_numbers(self, expression: str) -> list:
        """Extract numbers from expression, handling commas and various formats."""
        # Remove function names and parentheses
        cleaned = re.sub(r'[a-zA-Z_]+\(', '', expression)
        cleaned = cleaned.replace(')', '').replace(',', '')
        # Find all numbers (including decimals and negative)
        numbers = re.findall(r'-?\d+\.?\d*', cleaned)
        return [float(n) for n in numbers]

    def _calculate_growth_rate(self, expression: str) -> str:
        """Calculate percentage growth rate between two values."""
        numbers = self._extract_numbers(expression)
        if len(numbers) < 2:
            return "Error: growth_rate requires two values (old_value, new_value)"

        old_value, new_value = numbers[0], numbers[1]
        if old_value == 0:
            return "Error: Cannot calculate growth rate with zero as the starting value"

        growth = ((new_value - old_value) / abs(old_value)) * 100
        return f"Growth rate: {growth:.2f}% (from {old_value:,.2f} to {new_value:,.2f})"

    def _calculate_profit_margin(self, expression: str) -> str:
        """Calculate profit margin as percentage."""
        numbers = self._extract_numbers(expression)
        if len(numbers) < 2:
            return "Error: profit_margin requires two values (profit, revenue)"

        profit, revenue = numbers[0], numbers[1]
        if revenue == 0:
            return "Error: Cannot calculate profit margin with zero revenue"

        margin = (profit / revenue) * 100
        return f"Profit margin: {margin:.2f}% (profit: {profit:,.2f}, revenue: {revenue:,.2f})"

    def _calculate_percentage(self, expression: str) -> str:
        """Calculate what percentage part is of whole."""
        numbers = self._extract_numbers(expression)
        if len(numbers) < 2:
            return "Error: percentage requires two values (part, whole)"

        part, whole = numbers[0], numbers[1]
        if whole == 0:
            return "Error: Cannot calculate percentage with zero as the whole"

        pct = (part / whole) * 100
        return f"Percentage: {pct:.2f}% ({part:,.2f} out of {whole:,.2f})"

    def _calculate_ratio(self, expression: str) -> str:
        """Calculate simple ratio."""
        numbers = self._extract_numbers(expression)
        if len(numbers) < 2:
            return "Error: ratio requires two values (numerator, denominator)"

        numerator, denominator = numbers[0], numbers[1]
        if denominator == 0:
            return "Error: Cannot divide by zero"

        ratio = numerator / denominator
        return f"Ratio: {ratio:.4f} ({numerator:,.2f} / {denominator:,.2f})"

    def _safe_eval(self, expression: str) -> str:
        """Safely evaluate basic arithmetic expressions."""
        # Only allow numbers, basic operators, parentheses, and spaces
        allowed_chars = set('0123456789.+-*/() ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Expression contains invalid characters. Only basic arithmetic (+, -, *, /) is supported."

        try:
            # Use eval with restricted globals for safety
            result = eval(expression, {"__builtins__": {}}, {})
            if isinstance(result, (int, float)):
                return f"Result: {result:,.4f}" if isinstance(result, float) else f"Result: {result:,}"
            return f"Result: {result}"
        except Exception as e:
            return f"Error evaluating expression: {str(e)}"

    async def _arun(self, expression: str) -> str:
        """Async version of the tool."""
        return self._run(expression)
