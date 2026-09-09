import json
import os
import sys
import time
import re
import requests


# ============================================================
# AI YOUTUBE STORIES — SCENE PLANNER
# ============================================================

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

MODEL = "qwen2.5:7b"

SCENE_COUNT = 50

# Робимо менші batch-и для стабільності JSON.
BATCH_SIZE = 3

# Максимальний час одного запиту до Ollama.
TIMEOUT = 600

# Скільки разів повторювати невдалий batch.
MAX_RETRIES = 3

CHARACTER_FILE = "character.json"
SCENES_RAW_FILE = "scenes_raw.json"
SCENES_FILE = "scenes.json"


# ============================================================
# ПРОГРЕС
# ============================================================

class Progress:

    def __init__(self):
        self.start_time = time.time()

    def elapsed_seconds(self):
        return time.time() - self.start_time

    def elapsed(self):

        seconds = int(self.elapsed_seconds())

        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"

        return f"{minutes:02d}:{secs:02d}"

    def eta(self, completed, total):

        if completed <= 0:
            return "--:--"

        elapsed = self.elapsed_seconds()

        per_item = elapsed / completed

        remaining = int(
            per_item * (total - completed)
        )

        hours = remaining // 3600
        minutes = (remaining % 3600) // 60
        secs = remaining % 60

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"

        return f"{minutes:02d}:{secs:02d}"

    @staticmethod
    def bar(percent, width=30):

        percent = max(
            0,
            min(100, percent)
        )

        filled = int(
            width * percent / 100
        )

        return (
            "["
            + "█" * filled
            + "░" * (width - filled)
            + "]"
        )

    def show(
        self,
        overall_percent,
        stage,
        completed,
        total,
        extra=""
    ):

        # Очищення екрану Windows PowerShell.
        os.system("cls")

        print("=" * 64)
        print("          AI YouTube Stories — Scene Planner")
        print("=" * 64)
        print()

        print(
            f"ЗАГАЛЬНИЙ ПРОГРЕС: "
            f"{overall_percent:6.1f}%"
        )

        print(
            self.bar(overall_percent)
        )

        print()

        print(f"Етап: {stage}")

        if total > 0:

            stage_percent = (
                completed / total * 100
            )

        else:

            stage_percent = 0

        print(
            f"Прогрес етапу: "
            f"{completed}/{total} "
            f"({stage_percent:.1f}%)"
        )

        print(
            self.bar(stage_percent)
        )

        print()

        print(
            f"Минуло: {self.elapsed()}"
        )

        print(
            f"Орієнтовно залишилось: "
            f"{self.eta(completed, total)}"
        )

        if extra:

            print()
            print(extra)

        print()
        print("=" * 64)


# ============================================================
# JSON CLEANER
# ============================================================

def clean_json_text(text):

    if not text:
        raise ValueError(
            "Ollama повернув порожню відповідь."
        )

    text = text.strip()

    # --------------------------------------------------------
    # Прибираємо markdown ```json ... ```
    # --------------------------------------------------------

    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"^```\s*",
        "",
        text
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    text = text.strip()

    # --------------------------------------------------------
    # Іноді модель додає текст перед JSON.
    # Шукаємо першу {.
    # --------------------------------------------------------

    first_brace = text.find("{")

    if first_brace > 0:

        text = text[first_brace:]

    # --------------------------------------------------------
    # Іноді після JSON модель додає зайвий текст.
    # Беремо до останньої }.
    # --------------------------------------------------------

    last_brace = text.rfind("}")

    if last_brace >= 0:

        text = text[:last_brace + 1]

    return text.strip()


# ============================================================
# OLLAMA
# ============================================================

def ask_ollama(prompt):

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,

        # Ollama намагатиметься повернути JSON.
        "format": "json",

        "options": {
            "temperature": 0.35,

            # Трохи менший контекст,
            # щоб Qwen не починав плутатися.
            "num_ctx": 8192,

            # Не даємо відповідати безкінечно.
            "num_predict": 2500
        }
    }

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=TIMEOUT
        )

        response.raise_for_status()

    except requests.exceptions.ConnectionError:

        print()
        print("=" * 64)
        print("ПОМИЛКА: Ollama недоступний.")
        print("=" * 64)
        print()
        print(
            "Перевір:"
        )
        print(
            "http://127.0.0.1:11434"
        )
        print()

        raise

    except requests.exceptions.Timeout:

        print()
        print("=" * 64)
        print(
            f"ПОМИЛКА: Ollama не відповів "
            f"за {TIMEOUT} секунд."
        )
        print("=" * 64)
        print()

        raise

    response_data = response.json()

    raw_text = response_data.get(
        "response",
        ""
    )

    cleaned = clean_json_text(
        raw_text
    )

    try:

        return json.loads(cleaned)

    except json.JSONDecodeError as error:

        print()
        print("=" * 64)
        print("ПОМИЛКА: Qwen повернув пошкоджений JSON.")
        print("=" * 64)
        print()

        print(
            f"Рядок: {error.lineno}"
        )

        print(
            f"Колонка: {error.colno}"
        )

        print()

        print(
            "Відповідь моделі:"
        )

        print("-" * 64)

        print(
            raw_text[:5000]
        )

        print("-" * 64)
        print()

        raise ValueError(
            "Неправильний JSON від Ollama."
        )


# ============================================================
# ЗБЕРЕЖЕННЯ JSON
# ============================================================

def save_json(filename, data):

    temporary_file = filename + ".tmp"

    with open(
        temporary_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )

    # Атомарна заміна.
    os.replace(
        temporary_file,
        filename
    )


# ============================================================
# ЧИТАННЯ JSON
# ============================================================

def load_json(filename, default):

    if not os.path.exists(filename):

        return default

    try:

        with open(
            filename,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception as error:

        print()
        print(
            f"УВАГА: не вдалося прочитати "
            f"{filename}"
        )

        print(error)

        return default


# ============================================================
# CHARACTER
# ============================================================

def create_character(
    story_idea,
    progress
):

    if os.path.exists(
        CHARACTER_FILE
    ):

        character = load_json(
            CHARACTER_FILE,
            None
        )

        if character:

            progress.show(
                10,
                "Створення персонажа",
                1,
                1,
                (
                    f"Знайдено {CHARACTER_FILE}. "
                    f"Використовую існуючого героя."
                )
            )

            time.sleep(1)

            return character

    progress.show(
        0,
        "Створення персонажа",
        0,
        1,
        "Ollama створює Character Bible..."
    )

    prompt = f"""
Ти професійний сценарист.

Ти створюєш оригінальну історію для YouTube.

ІДЕЯ:

{story_idea}

Створи головного героя.

ВАЖЛИВО:

1. Історія повинна бути оригінальною.
2. Герой повинен мати стабільну зовнішність.
3. Не використовуй персонажів з відомих фільмів.
4. Не перекладай цей запит.
5. НЕ ПИШИ КИТАЙСЬКОЮ.
6. Відповідай ТІЛЬКИ JSON.
7. Ніякого тексту до або після JSON.

ФОРМАТ:

{{
  "name": "...",
  "age": 35,
  "gender": "male",
  "appearance": "...",
  "hair": "...",
  "face": "...",
  "body": "...",
  "clothing": "...",
  "personality": "...",
  "background": "..."
}}

Усі значення пиши українською.
"""

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            character = ask_ollama(
                prompt
            )

            if not isinstance(
                character,
                dict
            ):

                raise ValueError(
                    "Character не є JSON-об'єктом."
                )

            save_json(
                CHARACTER_FILE,
                character
            )

            progress.show(
                10,
                "Створення персонажа",
                1,
                1,
                "Character Bible збережено."
            )

            time.sleep(1)

            return character

        except Exception as error:

            print()
            print(
                f"Помилка Character "
                f"(спроба {attempt}/{MAX_RETRIES})"
            )
            print(error)

            if attempt < MAX_RETRIES:

                print(
                    "Повторюю через 3 секунди..."
                )

                time.sleep(3)

    raise RuntimeError(
        "Не вдалося створити Character Bible."
    )


# ============================================================
# СЦЕНИ
# ============================================================

def create_scene_batch(
    story_idea,
    character,
    start_id,
    end_id,
    previous_scenes
):

    count = (
        end_id - start_id + 1
    )

    previous_text = ""

    if previous_scenes:

        previous_text = (
            "\nОСТАННІ ПОПЕРЕДНІ СЦЕНИ:\n"
        )

        for scene in previous_scenes[-3:]:

            previous_text += (
                f"""
Сцена {scene.get("id")}:
{scene.get("title")}
{scene.get("description")}
"""
            )

    prompt = f"""
Ти професійний сценарист.

Ти пишеш ОРИГІНАЛЬНУ історію для YouTube.

ІДЕЯ ІСТОРІЇ:

{story_idea}

ГОЛОВНИЙ ГЕРОЙ:

{json.dumps(
    character,
    ensure_ascii=False,
    indent=2
)}

{previous_text}

ЗАРАЗ СТВОРИ РІВНО {count} СЦЕН.

Потрібні сцени:

від {start_id}
до {end_id}

ВАЖЛИВІ ПРАВИЛА:

1. Це одна безперервна історія.
2. Кожна нова сцена повинна продовжувати попередню.
3. Не повторюй попередні сцени.
4. Не перескакуй без пояснення через великі події.
5. Розвивай загадку поступово.
6. У кожній сцені повинна відбуватися конкретна подія.
7. Герой повинен залишатися тим самим персонажем.
8. Не використовуй відомі сюжети фільмів.
9. Історія повинна бути повністю оригінальною.
10. НЕ ПЕРЕКЛАДАЙ НІЧОГО КИТАЙСЬКОЮ.
11. НЕ ПИШИ КИТАЙСЬКОЮ МОВОЮ.
12. НЕ ПИШИ КОМЕНТАРІВ.
13. НЕ ДОДАВАЙ ТЕКСТ ДО JSON.
14. НЕ ДОДАВАЙ ТЕКСТ ПІСЛЯ JSON.
15. ПОВЕРНИ ТІЛЬКИ JSON.

ФОРМАТ ВІДПОВІДІ:

{{
  "scenes": [
    {{
      "id": {start_id},
      "title": "...",
      "description": "...",
      "location": "...",
      "time": "...",
      "action": "...",
      "camera": "...",
      "mood": "..."
    }}
  ]
}}

ПОВИННО БУТИ РІВНО {count} ЕЛЕМЕНТІВ.

ID ПОВИННІ БУТИ:

{start_id}, {start_id + 1}, ... {end_id}

Усі тексти пиши УКРАЇНСЬКОЮ.
"""

    result = ask_ollama(
        prompt
    )

    if not isinstance(
        result,
        dict
    ):

        raise ValueError(
            "Відповідь не є JSON-об'єктом."
        )

    if "scenes" not in result:

        raise ValueError(
            "У JSON немає поля scenes."
        )

    batch = result["scenes"]

    if not isinstance(
        batch,
        list
    ):

        raise ValueError(
            "scenes не є списком."
        )

    # --------------------------------------------------------
    # Перевіряємо кількість.
    # --------------------------------------------------------

    if len(batch) != count:

        raise ValueError(
            f"Ollama повернув {len(batch)} сцен "
            f"замість {count}."
        )

    # --------------------------------------------------------
    # Перевіряємо ID.
    # --------------------------------------------------------

    expected_ids = list(
        range(
            start_id,
            end_id + 1
        )
    )

    actual_ids = []

    for scene in batch:

        if not isinstance(
            scene,
            dict
        ):

            raise ValueError(
                "Одна зі сцен не є JSON-об'єктом."
            )

        if "id" not in scene:

            raise ValueError(
                "У сцени немає id."
            )

        actual_ids.append(
            scene["id"]
        )

    if actual_ids != expected_ids:

        raise ValueError(
            f"Неправильні ID сцен. "
            f"Очікувались {expected_ids}, "
            f"отримано {actual_ids}."
        )

    return batch


# ============================================================
# PROMPTS
# ============================================================

def create_prompt_batch(
    story_idea,
    character,
    scenes
):

    scenes_text = json.dumps(
        scenes,
        ensure_ascii=False,
        indent=2
    )

    prompt = f"""
Ти професійний prompt engineer
для Stable Diffusion 1.5.

ІДЕЯ:

{story_idea}

ГОЛОВНИЙ ГЕРОЙ:

{json.dumps(
    character,
    ensure_ascii=False,
    indent=2
)}

СЦЕНИ:

{scenes_text}

Створи для КОЖНОЇ сцени
англомовний prompt для Stable Diffusion 1.5.

Кожен prompt повинен містити:

- consistent character appearance
- clothing
- location
- action
- lighting
- atmosphere
- composition
- camera shot
- cinematic look
- realistic details

Головний герой повинен виглядати однаково
на всіх сценах.

НЕ ДОДАВАЙ:

- text
- letters
- words
- subtitles
- watermark
- logo

ВАЖЛИВО:

1. Prompts повинні бути АНГЛІЙСЬКОЮ.
2. Не пиши китайською.
3. Не перекладай запит.
4. Не додавай пояснення.
5. Поверни ТІЛЬКИ JSON.

ФОРМАТ:

{{
  "prompts": [
    {{
      "id": 1,
      "prompt": "cinematic realistic..."
    }}
  ]
}}
"""

    result = ask_ollama(
        prompt
    )

    if not isinstance(
        result,
        dict
    ):

        raise ValueError(
            "Prompts response не є JSON."
        )

    if "prompts" not in result:

        raise ValueError(
            "У відповіді немає prompts."
        )

    prompts = result["prompts"]

    if not isinstance(
        prompts,
        list
    ):

        raise ValueError(
            "prompts не є списком."
        )

    if len(prompts) != len(scenes):

        raise ValueError(
            "Кількість prompts не відповідає "
            "кількості сцен."
        )

    return prompts


# ============================================================
# ОСНОВНА ПРОГРАМА
# ============================================================

def main():

    if len(sys.argv) < 2:

        print()
        print(
            "Використання:"
        )
        print()
        print(
            'python scene_planner.py '
            '"Ідея історії"'
        )
        print()

        sys.exit(1)

    story_idea = sys.argv[1]

    progress = Progress()

    print()
    print(
        "AI YouTube Stories"
    )
    print()

    # ========================================================
    # ЕТАП 1 — CHARACTER
    # ========================================================

    character = create_character(
        story_idea,
        progress
    )

    # ========================================================
    # ЕТАП 2 — SCENES
    # ========================================================

    scenes = load_json(
        SCENES_RAW_FILE,
        []
    )

    if not isinstance(
        scenes,
        list
    ):

        scenes = []

    # --------------------------------------------------------
    # Очищаємо неправильні/дубльовані сцени.
    # --------------------------------------------------------

    valid_scenes = []

    seen_ids = set()

    for scene in scenes:

        if not isinstance(
            scene,
            dict
        ):

            continue

        scene_id = scene.get(
            "id"
        )

        if not isinstance(
            scene_id,
            int
        ):

            continue

        if scene_id in seen_ids:

            continue

        if scene_id < 1:

            continue

        if scene_id > SCENE_COUNT:

            continue

        valid_scenes.append(
            scene
        )

        seen_ids.add(
            scene_id
        )

    # --------------------------------------------------------
    # Сортуємо сцени.
    # --------------------------------------------------------

    valid_scenes.sort(
        key=lambda x: x["id"]
    )

    scenes = valid_scenes

    # --------------------------------------------------------
    # Визначаємо першу відсутню сцену.
    # --------------------------------------------------------

    completed_ids = {
        scene["id"]
        for scene in scenes
    }

    next_scene = 1

    while next_scene in completed_ids:

        next_scene += 1

    if next_scene > SCENE_COUNT:

        next_scene = SCENE_COUNT + 1

    # --------------------------------------------------------
    # Якщо вже є сцени — повідомляємо.
    # --------------------------------------------------------

    if scenes:

        print()
        print(
            f"Знайдено готових сцен: "
            f"{len(scenes)}/{SCENE_COUNT}"
        )

        print(
            f"Наступна сцена: "
            f"{next_scene}"
        )

        print()

        time.sleep(2)

    # ========================================================
    # ГЕНЕРАЦІЯ СЦЕН
    # ========================================================

    while len(completed_ids) < SCENE_COUNT:

        # ----------------------------------------------------
        # Визначаємо першу відсутню сцену.
        # ----------------------------------------------------

        start_id = 1

        while start_id in completed_ids:

            start_id += 1

        # ----------------------------------------------------
        # Створюємо batch.
        # ----------------------------------------------------

        end_id = min(
            start_id + BATCH_SIZE - 1,
            SCENE_COUNT
        )

        # ----------------------------------------------------
        # Якщо всередині діапазону є вже готові —
        # беремо тільки відсутні послідовні.
        # ----------------------------------------------------

        missing_ids = []

        for scene_id in range(
            start_id,
            end_id + 1
        ):

            if scene_id not in completed_ids:

                missing_ids.append(
                    scene_id
                )

            else:

                break

        if not missing_ids:

            continue

        start_id = missing_ids[0]
        end_id = missing_ids[-1]

        # ----------------------------------------------------
        # Batch номер.
        # ----------------------------------------------------

        batch_number = (
            (start_id - 1)
            // BATCH_SIZE
        ) + 1

        total_batches = (
            (
                SCENE_COUNT
                + BATCH_SIZE
                - 1
            )
            // BATCH_SIZE
        )

        # ----------------------------------------------------
        # Загальний прогрес.
        #
        # Character = 0–10%
        # Scenes   = 10–70%
        # Prompts  = 70–100%
        # ----------------------------------------------------

        overall = (
            10
            + (
                len(completed_ids)
                / SCENE_COUNT
            ) * 60
        )

        progress.show(
            overall,
            "Створення сцен",
            len(completed_ids),
            SCENE_COUNT,
            (
                f"Batch {batch_number}/"
                f"{total_batches} | "
                f"Створюємо сцени "
                f"{start_id}–{end_id}"
            )
        )

        # ----------------------------------------------------
        # Генерація з повторними спробами.
        # ----------------------------------------------------

        batch = None

        for attempt in range(
            1,
            MAX_RETRIES + 1
        ):

            try:

                print()
                print(
                    f"Запит до Ollama..."
                )

                print(
                    f"Спроба "
                    f"{attempt}/{MAX_RETRIES}"
                )

                batch = create_scene_batch(
                    story_idea,
                    character,
                    start_id,
                    end_id,
                    scenes
                )

                break

            except Exception as error:

                print()
                print(
                    "=" * 64
                )

                print(
                    f"Batch {start_id}–{end_id} "
                    f"не вдався."
                )

                print(
                    f"Причина: {error}"
                )

                print(
                    "=" * 64
                )

                if attempt < MAX_RETRIES:

                    print()
                    print(
                        "Повторюю batch через "
                        "5 секунд..."
                    )

                    time.sleep(5)

                else:

                    print()
                    print(
                        "Три спроби вичерпано."
                    )

                    print()
                    print(
                        "Вже готові сцени "
                        "ЗБЕРЕЖЕНІ."
                    )

                    print(
                        f"Готово: "
                        f"{len(completed_ids)}/"
                        f"{SCENE_COUNT}"
                    )

                    print()

                    raise RuntimeError(
                        f"Не вдалося створити "
                        f"сцени {start_id}–{end_id}."
                    )

        # ----------------------------------------------------
        # Додаємо batch.
        # ----------------------------------------------------

        if batch is None:

            raise RuntimeError(
                "Batch не створено."
            )

        scenes.extend(
            batch
        )

        # ----------------------------------------------------
        # Сортування.
        # ----------------------------------------------------

        scenes.sort(
            key=lambda x: x["id"]
        )

        # ----------------------------------------------------
        # Оновлюємо completed IDs.
        # ----------------------------------------------------

        completed_ids = {
            scene["id"]
            for scene in scenes
        }

        # ----------------------------------------------------
        # ОБОВ'ЯЗКОВО зберігаємо.
        # ----------------------------------------------------

        save_json(
            SCENES_RAW_FILE,
            scenes
        )

        overall = (
            10
            + (
                len(completed_ids)
                / SCENE_COUNT
            ) * 60
        )

        progress.show(
            overall,
            "Створення сцен",
            len(completed_ids),
            SCENE_COUNT,
            (
                f"Batch {start_id}–{end_id} "
                f"УСПІШНО ЗБЕРЕЖЕНО."
            )
        )

        time.sleep(1)

    # ========================================================
    # ВСІ СЦЕНИ
    # ========================================================

    print()
    print(
        "=" * 64
    )

    print(
        f"ВСІ {SCENE_COUNT} СЦЕН СТВОРЕНО."
    )

    print(
        f"Файл: {SCENES_RAW_FILE}"
    )

    print(
        "=" * 64
    )

    time.sleep(2)

    # ========================================================
    # ЕТАП 3 — PROMPTS
    # ========================================================

    # --------------------------------------------------------
    # Завантажуємо вже готові prompts.
    # --------------------------------------------------------

    existing_data = load_json(
        SCENES_FILE,
        []
    )

    existing_prompts = {}

    if isinstance(
        existing_data,
        list
    ):

        for item in existing_data:

            if not isinstance(
                item,
                dict
            ):

                continue

            scene_id = item.get(
                "id"
            )

            prompt = item.get(
                "prompt"
            )

            if (
                isinstance(scene_id, int)
                and isinstance(prompt, str)
                and prompt.strip()
            ):

                existing_prompts[
                    scene_id
                ] = prompt

    # --------------------------------------------------------
    # Генерація prompts.
    # --------------------------------------------------------

    completed_prompts = len(
        existing_prompts
    )

    while completed_prompts < SCENE_COUNT:

        missing_scenes = [
            scene
            for scene in scenes
            if scene["id"]
            not in existing_prompts
        ]

        if not missing_scenes:

            break

        batch_scenes = missing_scenes[
            :BATCH_SIZE
        ]

        start_id = batch_scenes[0]["id"]

        end_id = batch_scenes[-1]["id"]

        batch_number = (
            (start_id - 1)
            // BATCH_SIZE
        ) + 1

        total_batches = (
            (
                SCENE_COUNT
                + BATCH_SIZE
                - 1
            )
            // BATCH_SIZE
        )

        overall = (
            70
            + (
                completed_prompts
                / SCENE_COUNT
            ) * 30
        )

        progress.show(
            overall,
            "Створення prompts",
            completed_prompts,
            SCENE_COUNT,
            (
                f"Batch {batch_number}/"
                f"{total_batches} | "
                f"Prompts {start_id}–{end_id}"
            )
        )

        prompts = None

        # ----------------------------------------------------
        # Повторні спроби.
        # ----------------------------------------------------

        for attempt in range(
            1,
            MAX_RETRIES + 1
        ):

            try:

                print()
                print(
                    "Ollama створює prompts..."
                )

                print(
                    f"Спроба "
                    f"{attempt}/{MAX_RETRIES}"
                )

                prompts = create_prompt_batch(
                    story_idea,
                    character,
                    batch_scenes
                )

                break

            except Exception as error:

                print()
                print(
                    f"Помилка prompts "
                    f"(спроба {attempt}/"
                    f"{MAX_RETRIES})"
                )

                print(
                    error
                )

                if attempt < MAX_RETRIES:

                    print(
                        "Повтор через 5 секунд..."
                    )

                    time.sleep(5)

        if prompts is None:

            print()
            print(
                "Не вдалося створити prompts."
            )

            print(
                "Усі сцени вже збережені."
            )

            raise RuntimeError(
                f"Не вдалося створити "
                f"prompts для {start_id}–{end_id}."
            )

        # ----------------------------------------------------
        # Зберігаємо prompts.
        # ----------------------------------------------------

        for item in prompts:

            scene_id = item.get(
                "id"
            )

            prompt_text = item.get(
                "prompt"
            )

            if (
                isinstance(scene_id, int)
                and isinstance(prompt_text, str)
                and prompt_text.strip()
            ):

                existing_prompts[
                    scene_id
                ] = prompt_text.strip()

        completed_prompts = len(
            existing_prompts
        )

        # ----------------------------------------------------
        # Створюємо поточний scenes.json.
        # ----------------------------------------------------

        output_scenes = []

        for scene in scenes:

            scene_copy = dict(
                scene
            )

            scene_id = scene_copy[
                "id"
            ]

            if scene_id in existing_prompts:

                scene_copy[
                    "prompt"
                ] = existing_prompts[
                    scene_id
                ]

            output_scenes.append(
                scene_copy
            )

        save_json(
            SCENES_FILE,
            output_scenes
        )

        overall = (
            70
            + (
                completed_prompts
                / SCENE_COUNT
            ) * 30
        )

        progress.show(
            overall,
            "Створення prompts",
            completed_prompts,
            SCENE_COUNT,
            (
                f"Prompts {start_id}–{end_id} "
                f"УСПІШНО ЗБЕРЕЖЕНО."
            )
        )

        time.sleep(1)

    # ========================================================
    # ФІНАЛ
    # ========================================================

    final_scenes = []

    for scene in scenes:

        scene_copy = dict(
            scene
        )

        scene_id = scene_copy[
            "id"
        ]

        if scene_id in existing_prompts:

            scene_copy[
                "prompt"
            ] = existing_prompts[
                scene_id
            ]

        final_scenes.append(
            scene_copy
        )

    save_json(
        SCENES_FILE,
        final_scenes
    )

    progress.show(
        100,
        "ЗАВЕРШЕНО",
        SCENE_COUNT,
        SCENE_COUNT,
        "Усі сцени та prompts готові."
    )

    print()
    print(
        "=" * 64
    )

    print(
        "                         ГОТОВО!"
    )

    print(
        "=" * 64
    )

    print()

    print(
        f"Character: {CHARACTER_FILE}"
    )

    print(
        f"Scenes:    {SCENES_RAW_FILE}"
    )

    print(
        f"Final:     {SCENES_FILE}"
    )

    print()

    print(
        f"Час роботи: {progress.elapsed()}"
    )

    print()

    print(
        "Наступний етап:"
    )

    print(
        "Stable Diffusion → генерація зображень."
    )

    print()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print()
        print(
            "Програму зупинено користувачем."
        )

        print(
            "Вже збережені сцени залишилися "
            "у scenes_raw.json."
        )

        sys.exit(1)

    except Exception as error:

        print()
        print()
        print(
            "=" * 64
        )

        print(
            "ПРОГРАМА ЗУПИНЕНА"
        )

        print(
            "=" * 64
        )

        print()
        print(
            f"Причина: {error}"
        )

        print()
        print(
            "Якщо scenes_raw.json існує, "
            "вже готові сцени не втрачені."
        )

        print()

        sys.exit(1)