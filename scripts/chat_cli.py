#!/usr/bin/env python3
"""CLI Chatbot LawChat (RAG + LLM Ollama trực tiếp trên Terminal).

Cách dùng:
  1. Hỏi 1 câu:
     python scripts/chat_cli.py "Vượt đèn đỏ xe máy bị phạt bao nhiêu tiền?"

  2. Chế độ trò chuyện liên tục (Interactive REPL):
     python scripts/chat_cli.py
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid

# Đảm bảo import được frontend.api_client
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from frontend.api_client import APIError, LawChatAPI


def run_chat_session(
    api: LawChatAPI,
    username: str,
    password: str,
    one_shot_query: str | None = None,
) -> None:
    print("🔐 Đang đăng nhập tài khoản LawChat...")
    try:
        api.login(username, password)
    except APIError as exc:
        print(f"❌ Đăng nhập thất bại: {exc}")
        sys.exit(1)

    print(f"✅ Đăng nhập thành công với tài khoản: {username}")
    conv = api.create_conversation(title="CLI Chat Session")
    conv_id = conv["id"]

    def ask(prompt: str) -> None:
        print(f"\n💬 Người dùng: {prompt}")
        print("🤖 LawChat: ", end="", flush=True)
        client_msg_id = str(uuid.uuid4())
        citations = []
        limitations = []
        full_answer = []
        try:
            for event_type, payload in api.stream_chat(
                conversation_id=conv_id,
                client_message_id=client_msg_id,
                message=prompt,
            ):
                if event_type == "answer.delta":
                    delta = payload.get("text", "")
                    sys.stdout.write(delta)
                    sys.stdout.flush()
                    full_answer.append(delta)
                elif event_type == "citation":
                    cit = payload.get("citation")
                    if cit and cit not in citations:
                        citations.append(cit)
                elif event_type == "chat.completed":
                    res = payload.get("result", {})
                    if not full_answer and res.get("answer"):
                        sys.stdout.write(res["answer"])
                        sys.stdout.flush()
                    if not citations:
                        citations = res.get("citations", [])
                    limitations = res.get("limitations", [])
                elif event_type == "chat.failed":
                    print(f"\n❌ Lỗi phản hồi: {payload.get('message')}")
                    return

            print("\n")
            if citations:
                print("📚 Căn cứ pháp lý trích dẫn:")
                for cit in citations:
                    ref = cit.get("title", "")
                    doc_num = cit.get("document_number")
                    article = cit.get("article")
                    clause = cit.get("clause")
                    loc = []
                    if article:
                        loc.append(f"Điều {article}")
                    if clause:
                        loc.append(f"Khoản {clause}")
                    loc_str = f" ({', '.join(loc)})" if loc else ""
                    print(f"   • {ref} [{doc_num or 'VB'}]{loc_str}")
            if limitations:
                print("⚠️ Lưu ý:")
                for lim in limitations:
                    print(f"   • {lim}")
            print("─" * 70)
        except Exception as exc:
            print(f"\n❌ Lỗi trong quá trình phản hồi: {exc}")

    if one_shot_query:
        ask(one_shot_query)
        return

    # Chế độ Interactive REPL
    print("\n" + "=" * 70)
    print("🚀 ĐÃ KHỞI TẠO PHIÊN CHAT CLI THÀNH CÔNG (Gõ 'exit' hoặc 'quit' để thoát)")
    print("=" * 70)
    while True:
        try:
            prompt = input("\nNhập câu hỏi pháp lý: ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("exit", "quit", "q"):
                print("Tạm biệt!")
                break
            ask(prompt)
        except (KeyboardInterrupt, EOFError):
            print("\nĐã thoát phiên chat.")
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Chatbot LawChat trực tiếp trên CLI.")
    parser.add_argument("query", nargs="?", help="Câu hỏi pháp lý (bỏ trống để vào chế độ trò chuyện liên tục)")
    parser.add_argument("--username", default=os.getenv("LAWCHAT_CLI_USER", "testuser01"))
    parser.add_argument("--password", default=os.getenv("LAWCHAT_CLI_PASS", "testuser01"))
    parser.add_argument("--api-url", default=os.getenv("LAWCHAT_API_BASE_URL", "http://127.0.0.1:8000"))
    args = parser.parse_args()

    os.environ["LAWCHAT_API_BASE_URL"] = args.api_url
    api = LawChatAPI()
    try:
        run_chat_session(api, args.username, args.password, args.query)
    finally:
        api.close()


if __name__ == "__main__":
    main()
