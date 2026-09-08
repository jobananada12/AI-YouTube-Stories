import argparse

from core.llm import OllamaClient
from core.prompts import STORY_SYSTEM_PROMPT, story_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="AI YouTube Stories")
    parser.add_argument("topic", help="Seed or idea for the original story")
    parser.add_argument("--minutes", type=int, default=30)
    args = parser.parse_args()

    client = OllamaClient()
    result = client.generate(
        story_prompt(args.topic, args.minutes),
        system=STORY_SYSTEM_PROMPT,
    )
    print(result)


if __name__ == "__main__":
    main()
