import argparse
from datetime import datetime

from core.character_bible import CharacterBible
from core.project import StoryProject
from core.scene_planner import ScenePlanner
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

    print("Створюю Character Bible...")
    character_bible = CharacterBible().generate(story)
    print(f"Персонажів: {len(character_bible['characters'])}")

    script = ScriptWriter().write(story, args.minutes)
    print("Створюю режисерський план сцен...")
    scene_plan = ScenePlanner().plan(story, script, args.minutes)
    print(f"Сцен: {len(scene_plan.scenes)}")
    print(f"Орієнтовна тривалість: {scene_plan.total_duration_seconds // 60} хв")

    project_id = args.project or datetime.now().strftime("story_%Y%m%d_%H%M%S")
    project = StoryProject().create(story, script, project_id, scene_plan, character_bible)

    print(f"Готово. Проєкт збережено: {project}")
    print(f"Сценарій: {project / 'script.md'}")
    print(f"Character Bible: {project / 'character_bible.json'}")
    print(f"Режисерський план: {project / 'scenes.json'}")


if __name__ == "__main__":
    main()
