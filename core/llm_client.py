import base64
import json
import os
import uuid

import httpx

CLIENT_ID = os.getenv("GIGACHAT_CLIENT_ID", "019f2d7b-87f6-7eb2-95c9-3f46ab187df4")
CLIENT_SECRET = os.getenv("GIGACHAT_CLIENT_SECRET", "58152df2-e853-40be-832d-12ed703f8369")
SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
MODEL = "GigaChat"

_cached_token = None


async def get_giga_token() -> str:
    global _cached_token
    if _cached_token:
        return _cached_token

    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "RqUID": str(uuid.uuid4()),
    }

    auth_str = f"{CLIENT_ID}:{CLIENT_SECRET}"
    auth_bytes = auth_str.encode("utf-8")
    auth_b64 = base64.b64encode(auth_bytes).decode("utf-8")
    headers["Authorization"] = f"Basic {auth_b64}"

    payload = {"scope": SCOPE}

    async with httpx.AsyncClient(verify=False) as client:
        try:
            response = await client.post(url, headers=headers, data=payload, timeout=15.0)
            if response.status_code == 200:
                _cached_token = response.json().get("access_token")
                return _cached_token
        except Exception:
            pass
    return ""


async def ask_llm_json(system_prompt: str, user_text: str) -> str:
    token = await get_giga_token()
    if not token:
        return "{}"

    url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}],
        "temperature": 0.1,
        "max_tokens": 4000,
    }

    async with httpx.AsyncClient(verify=False) as client:
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=60.0)
            if response.status_code == 200:
                res_json = response.json()
                text = res_json["choices"][0]["message"]["content"]

                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()

                return text
        except Exception:
            pass
    return "{}"
