import argparse
import json

from src.query.qwen_query_compiler import (
    QwenQueryCompiler,
)

from src.query.query_planner import (
    QueryPlanner,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "query",
        type=str,
    )

    args = parser.parse_args()

    print(
        "\n============================"
    )

    print(
        "QUERY PLANNER"
    )

    print(
        "============================"
    )

    print(
        f"\nQuery: {args.query}"
    )

    #
    # Compile natural language
    #

    compiler = (
        QwenQueryCompiler()
    )

    compiled_query = (
        compiler.compile(
            args.query
        )
    )

    print(
        "\n--- COMPILED QUERY ---"
    )

    if hasattr(
        compiled_query,
        "model_dump",
    ):

        compiled_dict = (
            compiled_query.model_dump()
        )

    elif hasattr(
        compiled_query,
        "to_dict",
    ):

        compiled_dict = (
            compiled_query.to_dict()
        )

    else:

        compiled_dict = (
            compiled_query.__dict__
        )

    print(
        json.dumps(
            compiled_dict,
            indent=2,
            default=str,
        )
    )

    #
    # Build execution plan
    #

    planner = (
        QueryPlanner()
    )

    plan = (
        planner.plan(
            compiled_query
        )
    )

    print(
        "\n--- QUERY PLAN ---"
    )

    print(
        json.dumps(
            plan.to_dict(),
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()