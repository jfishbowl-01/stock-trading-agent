#!/usr/bin/env python3
import os
import sys
import argparse

# Import that works whether executed as module or script
try:
    # when run inside src/stock_analysis
    from .crew import StockAnalysisCrew  # type: ignore
except Exception:
    # when run from repo root or plain script
    from src.stock_analysis.crew import StockAnalysisCrew  # type: ignore


def run(ticker: str, question: str):
    """Kick off a one-shot analysis."""
    crew = StockAnalysisCrew(stock_symbol=ticker).crew()
    inputs = {
        "company_stock": ticker,
        "query": question,
    }
    return crew.kickoff(inputs=inputs)


def train(ticker: str, question: str, n_iterations: int):
    """Optional: train loop."""
    crew = StockAnalysisCrew(stock_symbol=ticker).crew()
    inputs = {
        "company_stock": ticker,
        "query": question,
    }
    crew.train(n_iterations=n_iterations, inputs=inputs)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="stock-analysis",
        description="Run or train the Stock Analysis Crew."
    )
    sub = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Run a single analysis (default).")
    p_run.add_argument("--ticker", default="AMZN", help="Stock ticker symbol (e.g., AAPL, IBM).")
    p_run.add_argument(
        "--query",
        default="Provide a concise investment thesis (<= 6 bullets) with verifiable sources.",
        help="User question/prompt."
    )

    p_train = sub.add_parser("train", help="Run training iterations.")
    p_train.add_argument("--ticker", default="AMZN", help="Stock ticker symbol.")
    p_train.add_argument(
        "--query",
        default="What was last year's revenue?",
        help="Training prompt."
    )
    p_train.add_argument("--iters", type=int, default=3, help="Number of training iterations.")

    args = parser.parse_args(argv)

    # Default to `run` if no subcommand
    cmd = args.cmd or "run"

    if cmd == "train":
        try:
            train(args.ticker.upper(), args.query, args.iters)
        except Exception as e:
            raise SystemExit(f"Training failed: {e}")
        return

    # run
    result = run(args.ticker.upper(), args.query)
    print("\n\n########################")
    print("## Here is the Report")
    print("########################\n")
    print(result)


if __name__ == "__main__":
    # Helpful banner
    print("## Welcome to Stock Analysis Crew")
    print("-------------------------------")
    main()