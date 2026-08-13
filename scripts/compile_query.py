import argparse
import json

from src.query.qwen_query_compiler import (
    QwenQueryCompiler,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "query",
        type=str,
    )

    args = parser.parse_args()

    compiler = (
        QwenQueryCompiler(
            device="cpu"
        )
    )

    compiled = (
        compiler.compile(
            args.query
        )
    )

    print(
        "\n============================"
    )

    print(
        "COMPILED VIDEO QUERY"
    )

    print(
        "============================\n"
    )

    print(
        json.dumps(
            compiled.to_dict(),
            indent=4,
        )
    )


if __name__ == "__main__":
    main()