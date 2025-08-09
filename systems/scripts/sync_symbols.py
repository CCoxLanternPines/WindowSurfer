if __name__ == "__main__":
    import argparse
    from systems.scripts.pair_cache import update_pair_cache

    parser = argparse.ArgumentParser(description="Refresh Kraken/Binance pair cache")
    parser.add_argument(
        "--out", default="data/tmp/pair_cache.json", help="Output path for cache JSON"
    )
    parser.add_argument(
        "--quotes",
        default="USD,USDT,USDC",
        help="Comma-separated preferred quotes order",
    )
    args = parser.parse_args()
    quotes = tuple([q.strip().upper() for q in args.quotes.split(",") if q.strip()])
    cache = update_pair_cache(out_path=args.out, preferred_quotes=quotes)
    print(f"[OK] Saved {len(cache)} bases to {args.out}")
