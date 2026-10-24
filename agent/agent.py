"""Консольный агент ЛР №2 — проект Watch Together.

Что здесь реализовано (шаги 4-6 ЛР, их разрешено делать с ИИ-ассистентом):
  * write_file с подтверждением [y/N] и описанием в TOOLS (шаг 4);
  * инструменты под проект: save_room_draft / list_room_drafts (шаг 5);
  * песочница agent/workspace/ и защита от выхода за её пределы.

Что оставлено тебе (шаги 1-3 — по правилам ЛР пишутся без ИИ-ассистента):
  * demo_first_request() — шаг 1: один запрос без инструментов;
  * demo_chat()          — шаг 2: чат с историей;
  * execute_tool()       — шаг 3: разбор JSON-аргументов и вызов инструмента;
  * run_agent()          — шаг 3: агентный цикл.

У каждого TODO в docstring есть точная спецификация — по ней код пишется
за 15-20 минут, а на защите его можно объяснить построчно.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Windows: при перенаправлении вывода Python берёт кодировку локали (cp1251),
# и печать ответа модели с символами вне cp1251 падает с UnicodeEncodeError.
# Принудительно переводим stdout/stderr в UTF-8.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE = BASE_DIR / "workspace"
ROOMS_DB = WORKSPACE / "rooms.json"

# .env лежит рядом с agent.py и в git не попадает (см. корневой .gitignore)
load_dotenv(BASE_DIR / ".env")

# Точный ID модели задаётся в .env — каталог: https://routerai.ru/api/v1/models
# Формат: провайдер/модель, например deepseek/deepseek-v4.1-flash-1017
MODEL = os.getenv("MODEL", "")
BASE_URL = os.getenv("BASE_URL", "https://routerai.ru/api/v1")
MAX_STEPS = int(os.getenv("MAX_STEPS", "6"))  # эксперимент 6.4: поставь 2

# Ключ RouterAI: https://routerai.ru (OpenRouter-совместимый API, но другой хост!)
API_KEY = (
    os.getenv("ROUTERAI_API_KEY")
    or os.getenv("OPENROUTER_API_KEY")  # поддержка старого имени переменной
    or os.getenv("API_KEY")
    or ""
)

# "not-set" нужен, чтобы импорт модуля не падал до проверки ключа в main():
# сам OpenAI() не принимает пустой api_key.
client = OpenAI(base_url=BASE_URL, api_key=API_KEY or "not-set")

SYSTEM_PROMPT = (
    "Ты — ассистент проекта Watch Together: сервис для совместного просмотра видео. "
    "Ты работаешь только внутри каталога workspace и помогаешь вести черновики комнат.\n"
    "Правила:\n"
    "1. Прежде чем говорить о содержимом файла — прочитай его инструментом.\n"
    "2. Не выдумывай содержимое файлов, которых не видел.\n"
    "3. Пути указывай относительно workspace, например README.md.\n"
    "4. Отвечай кратко, по делу, на русском языке."
)

# ---------------------------------------------------------------------------
# Песочница
# ---------------------------------------------------------------------------


def _resolve_path(raw_path: str) -> Path:
    """Переводит путь из аргумента модели в абсолютный путь внутри workspace.

    Это и есть защита песочницы: любой выход за пределы workspace (через ../
    или абсолютный путь) вызывает ValueError, а не открывает файл.
    """
    workspace_root = WORKSPACE.resolve()
    candidate = (workspace_root / raw_path).resolve()

    # Проверка вхождения: либо это сам каталог workspace, либо он среди родителей.
    if candidate != workspace_root and workspace_root not in candidate.parents:
        raise ValueError(f"путь вне песочницы workspace: {raw_path}")

    return candidate


# ---------------------------------------------------------------------------
# Базовые инструменты: файлы
# ---------------------------------------------------------------------------

MAX_READ_CHARS = 8000


def list_files() -> str:
    """Список файлов и каталогов в workspace."""
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    entries = sorted(WORKSPACE.iterdir(), key=lambda p: p.name)
    if not entries:
        return "workspace пуст"
    return "\n".join(f"[{'dir ' if p.is_dir() else 'file'}] {p.name}" for p in entries)


def read_file(path: str) -> str:
    """Читает текстовый файл из workspace."""
    try:
        target = _resolve_path(path)
    except ValueError as exc:
        return f"ОШИБКА: {exc}"

    if not target.exists():
        return f"ОШИБКА: файл не найден: {path}"
    if target.is_dir():
        return f"ОШИБКА: {path} — это каталог, используй list_files"

    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"ОШИБКА: не удалось прочитать {path}: {exc}"

    if len(text) > MAX_READ_CHARS:
        return text[:MAX_READ_CHARS] + f"\n... (обрезано, всего {len(text)} символов)"
    return text


def write_file(path: str, content: str) -> str:
    """Пишет файл в workspace с подтверждением пользователя.

    При отказе модель получает строку "ОТКАЗАНО: ..." — то есть видит, что
    запись не выполнена, и должна сообщить об этом в ответе.
    """
    try:
        target = _resolve_path(path)
    except ValueError as exc:
        return f"ОШИБКА: {exc}"

    preview = content if len(content) <= 2000 else content[:2000] + "\n... (обрезано в предпросмотре)"
    print(f"\n--- write_file: {path} ({len(content)} символов) ---")
    print(preview)
    print("--- конец файла ---")

    answer = input("Разрешить? [y/N] ").strip().lower()
    if answer not in ("y", "yes", "д", "да"):
        return f"ОТКАЗАНО: пользователь не разрешил запись в {path}"

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        return f"ОШИБКА: не удалось записать {path}: {exc}"

    return f"записано {len(content)} символов в {path}"


# ---------------------------------------------------------------------------
# Инструменты под проект: черновики комнат (FR-2.1 - FR-2.4 из README)
# ---------------------------------------------------------------------------

VISIBILITIES = ("public", "private")

PROVIDER_HOSTS = (
    ("youtube.com", "youtube"),
    ("youtu.be", "youtube"),
    ("vimeo.com", "vimeo"),
)


def _detect_provider(url: str) -> str:
    """Определяет провайдера видео по ссылке (FR-2.4)."""
    lowered = url.lower()
    for host, provider in PROVIDER_HOSTS:
        if host in lowered:
            return provider
    return "direct"


def _load_rooms() -> list[dict]:
    if not ROOMS_DB.exists():
        return []
    try:
        payload = json.loads(ROOMS_DB.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    rooms = payload.get("rooms", [])
    return rooms if isinstance(rooms, list) else []


def _save_rooms(rooms: list[dict]) -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    payload = {"rooms": rooms}
    ROOMS_DB.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_room_draft(name: str, video_url: str, visibility: str) -> str:
    """Сохраняет черновик комнаты в workspace/rooms.json."""
    name = (name or "").strip()
    if not name:
        return "ОШИБКА: название комнаты не может быть пустым"

    if not (video_url or "").lower().startswith(("http://", "https://")):
        return "ОШИБКА: ссылка на видео должна начинаться с http:// или https://"

    if visibility not in VISIBILITIES:
        return "ОШИБКА: visibility должна быть 'public' или 'private'"

    rooms = _load_rooms()
    draft = {
        "id": len(rooms) + 1,
        "name": name,
        "video_url": video_url,
        "provider": _detect_provider(video_url),
        "visibility": visibility,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    rooms.append(draft)

    try:
        _save_rooms(rooms)
    except OSError as exc:
        return f"ОШИБКА: не удалось сохранить rooms.json: {exc}"

    return f"черновик #{draft['id']} сохранён: {name} ({draft['provider']}, {visibility})"


def list_room_drafts(visibility: str = "all") -> str:
    """Показывает сохранённые черновики комнат."""
    if visibility not in ("all",) + VISIBILITIES:
        return "ОШИБКА: visibility должна быть 'all', 'public' или 'private'"

    rooms = _load_rooms()
    if visibility != "all":
        rooms = [room for room in rooms if room.get("visibility") == visibility]

    if not rooms:
        return "черновиков нет" if visibility == "all" else f"черновиков с visibility={visibility} нет"

    return "\n".join(
        f"#{room['id']} {room['name']} — {room['provider']}, "
        f"{room['visibility']}, {room['video_url']}"
        for room in rooms
    )


# ---------------------------------------------------------------------------
# Описания инструментов для модели
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "Показать файлы и каталоги в песочнице workspace. "
                "Вызывай первым, если не знаешь, какие файлы доступны."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Прочитать текстовый файл из workspace. "
                "Вызывай, когда нужно содержимое файла, а не догадка о нём."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Путь относительно workspace, например README.md",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Создать или перезаписать файл в workspace. "
                "Пользователь подтверждает каждую запись вручную, поэтому вызывай "
                "инструмент только когда действительно нужно сохранить результат."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Путь относительно workspace, например TODO.md",
                    },
                    "content": {
                        "type": "string",
                        "description": "Полное новое содержимое файла",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_room_draft",
            "description": (
                "Сохранить черновик комнаты совместного просмотра в workspace/rooms.json. "
                "Вызывай, когда пользователь просит создать комнату: указывает название "
                "и ссылку на видео."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Название комнаты",
                    },
                    "video_url": {
                        "type": "string",
                        "description": "Ссылка на видео: прямой URL, YouTube или Vimeo",
                    },
                    "visibility": {
                        "type": "string",
                        "enum": ["public", "private"],
                        "description": (
                            "public — вход по коду комнаты; "
                            "private — вход по коду и паролю комнаты"
                        ),
                    },
                },
                "required": ["name", "video_url", "visibility"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_room_drafts",
            "description": (
                "Показать сохранённые черновики комнат, при необходимости "
                "с фильтром по видимости."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "visibility": {
                        "type": "string",
                        "enum": ["all", "public", "private"],
                        "description": "Фильтр по видимости, по умолчанию all",
                    }
                },
                "required": [],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# ШАГ 3 — пишешь сам, без ИИ-ассистента
# ---------------------------------------------------------------------------


def execute_tool(name: str, raw_args: str) -> str:
    """Выполняет инструмент по имени и сырым JSON-аргументам.

    Спецификация из ЛР:
      1. json.loads(raw_args) — аргументы приходят строкой JSON.
      2. По name выбрать функцию: list_files / read_file / write_file /
         save_room_draft / list_room_drafts.
      3. Вернуть СТРОКУ — она попадёт в {"role": "tool", "content": ...}.
      4. Любое исключение (битый JSON, неизвестное имя, неправильные
         аргументы) перехватить и вернуть строкой, начинающейся с "ОШИБКА: ".
         Программа падать не должна.

    Пример:
        execute_tool("read_file", '{"path": "README.md"}')  -> текст README
        execute_tool("read_file", '{"path": "../.env"}')    -> "ОШИБКА: путь вне песочницы..."
    """
    try:
        args = json.loads(raw_args or "{}")          # 1
        dispatch = {"list_files": list_files, "read_file": read_file,
                    "write_file": write_file, "save_room_draft": save_room_draft,
                    "list_room_drafts": list_room_drafts}     # 2
        if name not in dispatch:                     # 3
            return f"ОШИБКА: неизвестный инструмент {name}"
        return dispatch[name](**args)                # 4
    except Exception as ex:                          # 5
        return f"ОШИБКА: {ex}"


def run_agent(messages: list[dict]) -> str:
    """Агентный цикл: запрос к модели -> инструменты -> снова запрос.

    Спецификация из ЛР:
      1. Не более MAX_STEPS итераций.
      2. На каждой итерации: client.chat.completions.create(
             model=MODEL, messages=messages, tools=TOOLS
         ).
      3. Печатать в лог: имя вызванного инструмента, его аргументы и usage
         (prompt_tokens / completion_tokens / total_tokens) — это пойдёт в отчёт.
      4. Ответ ассистента добавить в messages как есть (объект message
         с tool_calls тоже нужно сохранить в истории).
      5. Если tool_calls пустой — вернуть message.content и выйти из цикла.
      6. Иначе для каждого вызова:
             result = execute_tool(call.function.name, call.function.arguments)
             messages.append({
                 "role": "tool",
                 "tool_call_id": call.id,
                 "content": result,
             })
         и перейти к следующей итерации.
      7. Если MAX_STEPS исчерпан, вернуть понятное пользователю сообщение
         (например: "достигнут лимит шагов, задачу выполнить не удалось").
    """
    for step in range(MAX_STEPS):
        resp = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
        print(f"[шаг {step+1}] usage: {resp.usage.prompt_tokens}/{resp.usage.completion_tokens}")
        msg = resp.choices[0].message
        messages.append(msg)                    # объект целиком!
        if not msg.tool_calls:
            return msg.content
        for call in msg.tool_calls:
            print(f"  {call.function.name}({call.function.arguments})")
            result = execute_tool(call.function.name, call.function.arguments)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    return "достигнут лимит шагов, задача не выполнена"


# ---------------------------------------------------------------------------
# ШАГИ 1-2 — пишешь сам, без ИИ-ассистента
# ---------------------------------------------------------------------------


def demo_first_request() -> None:
    """Шаг 1: один запрос без инструментов.

    Напечатать три вещи:
      * response.choices[0].message.content
      * response.choices[0].finish_reason   (ожидается "stop")
      * response.usage                      (prompt / completion / total tokens)

    """
    content_text = 'Что такое токен?'

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {'role': 'user', 'content': content_text},
            ]
        )
    except Exception as ex:
        print(f'Возникла критическая ошибка при выполнении запроса: {ex}')
        return

    response_content = response.choices[0].message.content
    response_finish_reason = response.choices[0].finish_reason
    response_usage = response.usage

    print('Результат выполнения запроса:')
    print(f'content: {response_content}')
    print(f'finish_reason: {response_finish_reason}')
    print(f'usage: {response_usage}')

def demo_chat() -> None:
    """Шаг 2: чат с историей в цикле input().

    Требования:
      * messages начинается с системного промпта;
      * каждый вопрос пользователя добавляется в messages;
      * каждый ответ модели добавляется в messages;
      * выход по "exit"/"quit" или Ctrl+C.

    Проверка из ЛР: "Меня зовут ..." -> "Как меня зовут?" -> модель помнит.
    Затем закомментировать добавление истории и убедиться, что забыла.
    """
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    while True:
        add_content = input('Напечатайте запрос: ')
        if add_content in ('exit', 'quit'):
            print('Завершение работы программы')
            return

        messages.append({'role': 'user', 'content': add_content})

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
            )
        except Exception as ex:
            print(f'Возникла ошибка при выполнении запроса, повторите попытку: {ex}')
            continue

        if response and response.choices and response.choices[0].message:
            messages.append(response.choices[0].message)

        print(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

USAGE = """Использование:
  python agent.py ask                       шаг 1: один запрос без инструментов
  python agent.py chat                      шаг 2: чат с историей
  python agent.py run "прочитай README.md"  шаги 3-5: агентный цикл с инструментами

Команды ask/chat и цикл run реализуешь ты сам — см. TODO в этом файле.
"""


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(USAGE)
        return 1

    if not API_KEY:
        print("ОШИБКА: не задан ROUTERAI_API_KEY. Скопируй .env.example в .env и впиши ключ.")
        return 1

    if not MODEL:
        print(
            "ОШИБКА: не задан MODEL в .env.\n"
            "Впиши точный ID модели из каталога https://routerai.ru/api/v1/models "
            "(формат: провайдер/модель)."
        )
        return 1

    command = argv[1]
    args = argv[2:]

    if command == "ask":
        demo_first_request()
    elif command == "chat":
        demo_chat()
    elif command == "run":
        task = " ".join(args).strip()
        if not task:
            print("ОШИБКА: нужен текст задачи, например: python agent.py run \"прочитай README.md\"")
            return 1
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]
        print(run_agent(messages))
    else:
        print(USAGE)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
