# Financial Analysis Agent

An intelligent agent for analyzing SEC 10-K filings using RAG-based document retrieval and LangChain.

## Overview

This project enables natural language queries over SEC Form 10-K financial filings. The system combines a large language model with a vector database to retrieve relevant document fragments and perform financial analysis.

**Key Capabilities:**
- Search and extract information from SEC 10-K filings
- Perform financial calculations (growth rates, margins, ratios)
- Multi-step reasoning with transparent tool usage
- Interactive web-based chat interface

## Architecture

```
Streamlit UI → LangChain Agent → Tools
                                   ├── RAG Search (ChromaDB + HuggingFace embeddings)
                                   └── Financial Calculator
```

**Tech Stack:**
- **LangChain** – Agent orchestration and ReAct-style reasoning
- **OpenAI gpt-4o-mini** – LLM for response generation
- **ChromaDB** – Persistent vector store
- **HuggingFace** – Sentence embeddings (`all-mpnet-base-v2`)
- **Streamlit** – Web interface

## Dataset

SEC 10-K filings from 10 S&P 500 companies (2010–2019):
MCHP, MAR, SBUX, VRSK, MSI, A, STT, PH, SRE, SPG

## Installation

**Prerequisites:** Python 3.9+, OpenAI API key

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

## Usage

```bash
streamlit run app.py
```

The application opens at `http://localhost:8501`

## Example Queries

| Query | Description |
|-------|-------------|
| "What was Starbucks' revenue in 2019?" | Single document lookup |
| "Compare Starbucks' revenue between 2018 and 2019" | Multi-step retrieval + calculation |
| "Calculate growth rate from $10M to $15M" | Direct financial calculation |

## Project Structure

```
├── app.py                      # Streamlit interface
├── agent.py                    # LangChain agent definition
├── system.py                   # RAG retrieval system
├── config.py                   # Configuration settings
├── tools/
│   ├── rag_tool.py             # Document search tool
│   └── calculator_tool.py      # Financial calculator
├── data/
│   ├── original/               # Raw SEC filings
│   ├── processed/              # Cleaned documents
│   └── vector_store/           # ChromaDB storage
└── data_processing.py          # Document preprocessing
```

## Limitations

- Data limited to 2010–2019 filings for 10 companies
- Only 10-K filings (no 10-Q or 8-K)
- Basic financial calculations only
- No real-time market data

## License

MIT
