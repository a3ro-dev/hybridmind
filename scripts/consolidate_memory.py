"""
CLI script: consolidate old session memories into summary nodes.

Usage:
  python scripts/consolidate_memory.py [--min-facts 5] [--max-age-hours 24]

Requires HybridMind server dependencies and a provider allowed by config.py.
Hack Club is used only when HYBRIDMIND_ALLOW_RESEARCH_PROXY=true.
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Consolidate old session memories")
    parser.add_argument("--min-facts", type=int, default=5,
                        help="Min facts per session to trigger consolidation (default: 5)")
    parser.add_argument("--max-age-hours", type=int, default=24,
                        help="Only consolidate sessions older than N hours (default: 24)")
    parser.add_argument("--model", type=str, default=None,
                        help="LLM model for summarization (default: HYBRIDMIND_CONSOLIDATION_MODEL env)")
    args = parser.parse_args()

    logger.info("Initializing HybridMind storage...")
    from api.dependencies import get_db_manager
    db_manager = get_db_manager()

    logger.info(
        f"Running consolidation: min_facts={args.min_facts}, "
        f"max_age_hours={args.max_age_hours}"
    )
    from engine.consolidation import consolidate_sessions
    result = consolidate_sessions(
        db_manager,
        min_facts=args.min_facts,
        max_age_hours=args.max_age_hours,
        model=args.model,
    )

    print(f"\nConsolidation complete:")
    print(f"  Sessions found:      {result.get('sessions_total', 0)}")
    print(f"  Sessions processed:  {result.get('sessions_processed', 0)}")
    print(f"  Summary nodes created: {result.get('summaries_created', 0)}")
    if result.get("error"):
        print(f"  Error: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
