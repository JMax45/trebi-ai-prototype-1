import httpx
import json
import sys

BASE_URL = "http://localhost:8000"
SESSION_ID = "default"

def clear_session():
    try:
        r = httpx.delete(f"{BASE_URL}/session/{SESSION_ID}", timeout=10.0)
        print("[contesto cancellato]")
    except Exception as e:
        print(f"[errore: {e}]")

def stream_question(question: str):
    with httpx.stream(
        "POST",
        f"{BASE_URL}/chat/stream",
        json={"question": question, "session_id": SESSION_ID},
        timeout=120.0,
    ) as r:
        source = None
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if "source" in data:
                source = data["source"]
                #print(f"\n[fonte: {source}]\n", flush=True)
            elif "tool_call" in data:
                args_preview = ", ".join(f"{k}={repr(v)[:40]}" for k, v in data.get("args", {}).items())
                print(f"\n[tool: {data['tool_call']}({args_preview})]", flush=True)
            elif "token" in data:
                print(data["token"], end="", flush=True)
            elif "saved" in data:
                print(f"\n[salvato: {data['saved']}]", flush=True)
            elif "error" in data:
                print(f"\n[errore: {data['error']}]", flush=True)
        print()  # newline after response

def main():
    print("Chat RAG — digita 'exit' per uscire.\n")
    while True:
        try:
            question = input("Tu: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nArrivederci!")
            sys.exit(0)

        if not question:
            continue
        if question.lower() in ("exit", "quit", "esci"):
            print("Arrivederci!")
            sys.exit(0)
        if question.lower() in ("/clear", "clear", "reset"):
            clear_session()
            continue

        print("AI: ", end="", flush=True)
        try:
            stream_question(question)
        except httpx.ConnectError:
            print(f"[errore: impossibile connettersi a {BASE_URL} — il server è in esecuzione?]")
        except Exception as e:
            print(f"[errore: {e}]")

if __name__ == "__main__":
    main()
