"""
chat.py — Interactive real-time CLI for the Memory Compression Engine.

Type sentences to store them as compressed memories; ask questions to
retrieve them. Pure standard library — nothing to install.

Run:  python chat.py
"""
from engine import MemoryStore

HELP = """
Commands:
  <just type anything>   Store it as a new memory (auto-compressed & tiered)
  /ask <question>        Retrieve the most relevant memories for a question
  /list                  Show all stored memories and their tiers
  /stats                 Show compression statistics
  /save <file>           Save memories to a text file
  /load <file>           Load memories from a text file
  /help                  Show this help
  /quit                  Exit
"""


def print_memories(store):
    if not store.memories:
        print("  (no memories yet)")
        return
    for i, m in enumerate(store.memories):
        print(f"  [{i}] ({m.tier:5}) imp={m.importance:.2f} "
              f"ratio={m.compression_ratio:.2f}")
        print(f"       {m.text}")


def save_memories(store, path):
    with open(path, "w", encoding="utf-8") as f:
        for m in store.memories:
            # save the ORIGINAL source so it can be re-compressed on load
            f.write(m.source_text.replace("\n", " ") + "\n")
    print(f"  Saved {len(store.memories)} memories to {path}")


def load_memories(store, path):
    import os
    if not os.path.exists(path):
        print(f"  File not found: {path}")
        return
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    for ln in lines:
        store.add(ln)
    print(f"  Loaded {len(lines)} memories from {path}")


def chat_loop():
    print("=" * 62)
    print(" AI Digital Memory Compression Engine — REAL-TIME MODE")
    print("=" * 62)
    print("Type anything to store it as a memory. Ask with  /ask <question>")
    print("Type  /help  for all commands,  /quit  to exit.\n")

    store = MemoryStore(raw_window=2, light_window=3)

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user:
            continue

        # ---- commands ----
        if user.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Goodbye.")
            break

        elif user.lower() in ("/help", "help", "?"):
            print(HELP)

        elif user.lower() == "/list":
            print_memories(store)

        elif user.lower() == "/stats":
            for k, v in store.stats().items():
                print(f"  {k}: {v}")

        elif user.lower().startswith("/ask"):
            q = user[4:].strip()
            if not q:
                print("  Usage: /ask <your question>")
                continue
            hits = store.retrieve(q, k=3)
            if not hits:
                print("  No memories stored yet.")
                continue
            print(f"  Top memories for: \"{q}\"")
            for m, s in hits:
                print(f"    ({s:.3f}) [{m.tier}] {m.text}")

        elif user.lower().startswith("/save"):
            path = user[5:].strip() or "memories.txt"
            save_memories(store, path)

        elif user.lower().startswith("/load"):
            path = user[5:].strip() or "memories.txt"
            load_memories(store, path)

        elif user.startswith("/"):
            print("  Unknown command. Type /help.")

        # ---- plain text => store as memory ----
        else:
            store.add(user)
            newest = store.memories[-1]
            print(f"  stored (tier={newest.tier}, importance={newest.importance:.2f}, "
                  f"{len(store.memories)} total)")



if __name__ == "__main__":
    chat_loop()
