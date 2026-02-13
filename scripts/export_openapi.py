"""Export OpenAPI spec from the FastAPI app without running the server.

Usage:
    uv run python scripts/export_openapi.py              # JSON to docs/openapi.json
    uv run python scripts/export_openapi.py --format yaml # YAML to docs/openapi.yaml
    uv run python scripts/export_openapi.py -o /tmp/spec.json
    uv run python scripts/export_openapi.py --stats       # just print endpoint count and size
"""
import argparse
import sys
from pathlib import Path

import orjson


def get_schema() -> dict:
    """Import the FastAPI app and extract the OpenAPI schema."""
    from heylook_llm.api import app
    return app.openapi()


def write_json(schema: dict, path: Path) -> int:
    data = orjson.dumps(schema, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
    path.write_bytes(data)
    return len(data)


def write_yaml(schema: dict, path: Path) -> int:
    try:
        import yaml
    except ImportError:
        print("PyYAML not installed. Run: uv add pyyaml", file=sys.stderr)
        sys.exit(1)
    text = yaml.dump(schema, default_flow_style=False, sort_keys=True, allow_unicode=True)
    path.write_text(text)
    return len(text.encode())


def print_stats(schema: dict) -> None:
    paths = schema.get("paths", {})
    endpoint_count = 0
    by_tag: dict[str, list[str]] = {}

    for path, methods in paths.items():
        for method, details in methods.items():
            if method in ("get", "post", "put", "delete", "patch"):
                endpoint_count += 1
                tags = details.get("tags", ["Untagged"])
                for tag in tags:
                    by_tag.setdefault(tag, []).append(f"{method.upper():6s} {path}")

    raw_size = len(orjson.dumps(schema))
    print(f"Endpoints: {endpoint_count}")
    print(f"Spec size: {raw_size:,} bytes ({raw_size / 1024:.1f} KB)")
    print(f"Schemas:   {len(schema.get('components', {}).get('schemas', {}))}")
    print()
    for tag in sorted(by_tag):
        print(f"[{tag}]")
        for ep in sorted(by_tag[tag]):
            print(f"  {ep}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Export OpenAPI spec from HeylookLLM")
    parser.add_argument("-o", "--output", type=Path, help="Output file path")
    parser.add_argument("--format", choices=["json", "yaml"], default="json", help="Output format (default: json)")
    parser.add_argument("--stats", action="store_true", help="Print spec stats and exit")
    args = parser.parse_args()

    schema = get_schema()

    if args.stats:
        print_stats(schema)
        return

    out_path = args.output or Path("docs") / f"openapi.{args.format}"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "yaml":
        size = write_yaml(schema, out_path)
    else:
        size = write_json(schema, out_path)

    print(f"Wrote {size:,} bytes to {out_path}")
    print_stats(schema)


if __name__ == "__main__":
    main()
