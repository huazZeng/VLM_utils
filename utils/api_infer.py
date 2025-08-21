#!/usr/bin/env python3
from typing import List, Dict, Optional
from openai import AsyncOpenAI
import asyncio

class OpenAIChatClient:
    """
    Thin wrapper over OpenAI-compatible chat.completions API using AsyncOpenAI.
    No preprocessing or postprocessing. You must pass messages in the OpenAI format.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        try:
            self.async_client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
        except Exception as e:
            print(f"Failed to initialize AsyncOpenAI client: {e}")
            raise

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        skip_special_tokens: bool = True,
    ) -> str:
        try:
            resp = await self.async_client.completions.create(
                model=self.model_name,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                skip_special_tokens=True,
            )
            return resp.choices[0].text
        except Exception as e:
            print(f"Error in complete: {e}")
            return ""

    async def chat(
        self,
        messages: List[Dict],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        try:
            resp = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"Error in chat: {e}")
            return ""

    async def chat_many(
        self,
        messages_list: List[List[Dict]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        concurrency: int = 8,
    ) -> List[str]:
        try:
            import asyncio

            semaphore = asyncio.Semaphore(concurrency)

            async def _one(msgs: List[Dict]):
                try:
                    async with semaphore:
                        return await self.chat(msgs, temperature=temperature, max_tokens=max_tokens)
                except Exception as e:
                    print(f"Error in _one: {e}")
                    return ""

            return await asyncio.gather(*[_one(msgs) for msgs in messages_list])
        except Exception as e:
            print(f"Error in chat_many: {e}")
            return [""] * len(messages_list)

if __name__ == "__main__":
    client = OpenAIChatClient(
        base_url="https://api.boyuerichdata.opensphereai.com/v1",
        api_key="sk-rc13fDLPmFGMysHxbLHwS18RFFGxEBZTju6eNpgTDMxt8kkG",
        model_name="gemini-2.5-pro",
    )
    print(asyncio.run(client.chat([{"role": "user", "content": "Hello, how are you?"}])))