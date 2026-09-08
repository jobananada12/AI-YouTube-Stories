import argparse
from datetime import datetime

from core.project import StoryProject
from core.script_writer import ScriptWriter
from core.story_generator import StoryGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="AI YouTube Stories")
    parser.add_argument("topic", help="Seed or idea for the original story")
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--project", default=None, help="Project ID; generated automatically when omitted")
    args = parser.parse_args()

    if args.minutes < 10 or args.minutes > 120:
        parser.error("--minutes must be between 10 and 120")

    story = StoryGenerator().generate(args.topic, args.minutes)
    print(f"\nІСТОРІЯ: {story.title}")
    print(f"ЖАНР: {story.genre}")
    print(f"ЛОГЛАЙН: {story.logline}\n")

    script = ScriptWriter().write(story, args.minutes)
    project_id = args.project or datetime.now().strftime("story_%Y%m%d_%H%M%S")
    project = StoryProject().create(story, script, project_id)

    print(f"Готово. Проєкт збережено: {project}")
    print(f"Сценарій: {project / 'script.md'}")


if __name__ == "__main__":
    main()
