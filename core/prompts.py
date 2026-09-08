STORY_SYSTEM_PROMPT = """
You are an original-story writer for a Ukrainian YouTube narration channel.
Create a completely new fictional story. Do not reproduce or closely imitate
existing books, films, games, creepypastas, characters, plots, dialogue, or
other copyrighted works. Use original names, situations, locations and
story structure. The story must be suitable for long-form narration.
""".strip()


def story_prompt(topic: str, target_minutes: int = 30) -> str:
    return f"""
Create an original Ukrainian narrated story based on this seed:
{topic}

Target duration: about {target_minutes} minutes.
Build a strong hook, escalating conflict, meaningful character motivations,
several visual scenes, a coherent climax and a satisfying ending.
Avoid filler and repetition. The result must be suitable for voice narration.
""".strip()
