import pandas as pd
import random


STUDENT_ID: int = 904061372  # TODO
NUM_DOCS: int = 10


def get_sp500_tickers_wikipedia() -> list[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    sp500_table = tables[0]
    return sp500_table["Symbol"].tolist()


all_sp500_tickers = get_sp500_tickers_wikipedia()

random.seed(STUDENT_ID)
sampled_tickers = sorted(random.sample(all_sp500_tickers, NUM_DOCS))

with open("sampled_tickers.txt", "w") as f:
    f.write("\n".join(sampled_tickers))