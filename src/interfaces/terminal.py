from __future__ import annotations

from src.interfaces.mcp_client import MCPBackend


class TerminalApp:
    """Интерактивный терминал для поиска."""

    def __init__(self, backend: MCPBackend, title: str):
        self.backend = backend
        self.title = title

    def run(self) -> None:
        print(f"{self.title}. Введите /exit для выхода.")
        print("Команды: /provider, /model, /status, /help")

        try:
            while True:
                user_input = input("\n> ").strip()
                if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
                    print("Выход.")
                    break
                if user_input.startswith("/"):
                    if self._handle_command(user_input):
                        continue
                if not user_input:
                    continue

                print("Обработка запроса...")
                try:
                    result = self.backend.search(user_input, limit=5)
                    print("\nОтвет:\n", result)
                except Exception as exc:
                    print(f"Ошибка: {exc}")
        except KeyboardInterrupt:
            print("\nВыход.")
        finally:
            self.backend.close()

    def _handle_command(self, command: str) -> bool:
        parts = command.strip().split()
        cmd = parts[0].lower()

        if cmd == "/help":
            self._print_help()
            return True

        if cmd == "/provider":
            if len(parts) == 1:
                print(self.backend.llm_status())
                return True
            provider = parts[1]
            model = parts[2] if len(parts) > 2 else None
            print(self.backend.llm_set(provider, model=model))
            return True

        if cmd == "/model":
            if len(parts) < 2:
                print("Укажите имя модели. Пример: /model qwen2.5:7b или /model openai/gpt-4o-mini")
                return True
            model = " ".join(parts[1:]).strip()
            print(self.backend.llm_model(model))
            return True

        if cmd == "/status":
            print(self.backend.llm_status())
            return True

        return False

    @staticmethod
    def _print_help() -> None:
        print("Команды:")
        print("  /provider               - показать текущий провайдер")
        print("  /provider <name>        - переключить (direct, ollama или openrouter)")
        print("  /provider <name> <model> - переключить и задать модель")
        print("  /model <name>           - сменить модель текущего провайдера")
        print("  /status                 - показать конфигурацию")
        print("  /exit                   - выход")
