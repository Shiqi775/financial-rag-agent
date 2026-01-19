import streamlit as st
from system import RAGSystem
from agent import create_agent
import traceback

st.set_page_config(
    page_title="Financial Analysis Agent",
    page_icon="📊",
    layout="wide"
)


def load_tickers():
    """Load available tickers from file."""
    try:
        with open("sampled_tickers.txt", 'r') as file:
            return [line.strip() for line in file.readlines()]
    except Exception as e:
        st.error(f"Error loading tickers: {str(e)}")
        return []


def initialize_systems():
    """Initialize RAG system and agent."""
    try:
        if 'rag_system' not in st.session_state:
            with st.spinner("Initializing RAG system..."):
                st.session_state.rag_system = RAGSystem()

        if 'agent' not in st.session_state:
            with st.spinner("Initializing AI agent..."):
                st.session_state.agent = create_agent(
                    st.session_state.rag_system,
                    verbose=False
                )

        if 'messages' not in st.session_state:
            st.session_state.messages = []

        return True
    except Exception as e:
        st.error(f"Failed to initialize systems: {str(e)}")
        st.error("Please check your configuration and API keys.")
        return False


def display_tool_usage(steps):
    """Display agent's tool usage in an expandable section."""
    if not steps:
        return

    with st.expander("🔧 Agent Reasoning & Tool Usage", expanded=False):
        for i, step in enumerate(steps, 1):
            st.markdown(f"**Step {i}: {step['tool']}**")

            # Display tool input
            st.markdown("*Input:*")
            if isinstance(step['tool_input'], dict):
                for key, value in step['tool_input'].items():
                    st.markdown(f"- `{key}`: {value}")
            else:
                st.code(str(step['tool_input']), language=None)

            # Display tool output (truncated)
            st.markdown("*Output:*")
            output = str(step['observation'])
            if len(output) > 500:
                output = output[:500] + "..."
            st.code(output, language=None)

            if i < len(steps):
                st.markdown("---")


def main():
    st.title("📊 Financial Analysis Agent")
    st.markdown("*Intelligent assistant for SEC 10-K financial document analysis*")

    # Initialize systems
    if not initialize_systems():
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This agent can:
        - 🔍 Search SEC 10-K filings
        - 📈 Perform financial calculations
        - 💬 Answer questions about companies

        **Available Data:**
        - Years: 2010-2019
        - Companies: 10 tickers
        """)

        # Show available tickers
        tickers = load_tickers()
        if tickers:
            st.markdown("**Available Tickers:**")
            st.markdown(", ".join(tickers))

        st.markdown("---")

        # Optional filters
        st.header("🎯 Optional Filters")
        st.markdown("*The agent can infer these, but you can specify if desired.*")

        ticker_filter = st.selectbox(
            "Filter by Ticker",
            options=["(Let agent decide)"] + tickers,
            index=0
        )

        year_filter = st.selectbox(
            "Filter by Year",
            options=["(Let agent decide)"] + [str(y) for y in range(2010, 2020)],
            index=0
        )

        st.markdown("---")

        # Clear conversation button
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.session_state.agent.clear_history()
            st.rerun()

        # System status
        st.markdown("### System Status")
        st.markdown(f"✅ RAG System: Active")
        st.markdown(f"✅ Agent: Ready")
        st.markdown(f"🖥️ Device: {st.session_state.rag_system.device}")

    # Main chat area
    st.markdown("### 💬 Chat")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "tool_usage" in message and message["tool_usage"]:
                display_tool_usage(message["tool_usage"])

    # Chat input
    if prompt := st.chat_input("Ask about financial documents..."):
        # Add context from filters if specified
        enhanced_prompt = prompt
        if ticker_filter != "(Let agent decide)" or year_filter != "(Let agent decide)":
            context_parts = []
            if ticker_filter != "(Let agent decide)":
                context_parts.append(f"ticker: {ticker_filter}")
            if year_filter != "(Let agent decide)":
                context_parts.append(f"year: {year_filter}")
            if context_parts:
                enhanced_prompt = f"{prompt} (Context: {', '.join(context_parts)})"

        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.agent.invoke(enhanced_prompt)

                    # Display response
                    st.markdown(result["output"])

                    # Display tool usage
                    if result["intermediate_steps"]:
                        display_tool_usage(result["intermediate_steps"])

                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["output"],
                        "tool_usage": result["intermediate_steps"]
                    })

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    if st.checkbox("Show detailed error"):
                        st.code(traceback.format_exc())

    # Example queries
    with st.expander("📝 Example Queries"):
        st.markdown("""
        **Basic Queries:**
        - What was Starbucks' revenue in 2019?
        - Tell me about Starbucks's risk factors in 2018.
        - What are the main business segments of SBUX?

        **Calculations:**
        - Calculate the growth rate if revenue went from \$10 million to \$15 million.
        - What's the profit margin if profit is \$2M and revenue is \$10M?

        **Comparisons:**
        - Compare the revenue between 2018 and 2019 for Starbucks.
        """)


if __name__ == "__main__":
    main()
