#!/usr/bin/env python3
from typing import List, Dict, Optional
from openai import AsyncOpenAI


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
        self.async_client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        skip_special_tokens: bool = True,
    ) -> str:
        resp = await self.async_client.completions.create(
            model=self.model_name,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            skip_special_tokens=True,
        )
        

        return resp.choices[0].text
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
        except Exception as e:
            print(e)
        return resp.choices[0].message.content

    async def chat_many(
        self,
        messages_list: List[List[Dict]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        concurrency: int = 8,
    ) -> List[str]:
        import asyncio

        semaphore = asyncio.Semaphore(concurrency)

        async def _one(msgs: List[Dict]):
            async with semaphore:
                return await self.chat(msgs, temperature=temperature, max_tokens=max_tokens)

        return await asyncio.gather(*[_one(msgs) for msgs in messages_list]) 