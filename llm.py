"""
Shared Azure OpenAI client factory.
All patterns import from here — change deployments in one place.
"""

import os
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings


def get_llm(
    deployment: str = "gpt-4o",
    temperature: float = 0,
    streaming: bool = False,
    max_tokens: int = 4096,
    callbacks: list | None = None,
) -> AzureChatOpenAI:
    kwargs = dict(
        azure_deployment=deployment,
        api_version="2024-10-21",
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
        # Reads: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY
    )
    if callbacks:
        kwargs["callbacks"] = callbacks
    return AzureChatOpenAI(**kwargs)


def get_cheap_llm(**kwargs) -> AzureChatOpenAI:
    return get_llm(deployment="gpt-4o-mini", **kwargs)


def get_embeddings(deployment: str = "text-embedding-3-large") -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_deployment=deployment,
        # Reads: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY
    )


# Required env vars — validated on import in production
REQUIRED_ENV_VARS = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
]

OPTIONAL_ENV_VARS = {
    "AZURE_SEARCH_ENDPOINT": "Pattern 03 — Azure AI Search",
    "AZURE_SEARCH_API_KEY": "Pattern 03 — Azure AI Search",
    "LANGFUSE_PUBLIC_KEY": "Pattern 04 — LangFuse observability",
    "LANGFUSE_SECRET_KEY": "Pattern 04 — LangFuse observability",
    "LANGFUSE_HOST": "Pattern 04 — LangFuse host (default: cloud.langfuse.com)",
}


def check_env(require_all: bool = False) -> dict[str, bool]:
    results = {}
    for var in REQUIRED_ENV_VARS:
        results[var] = bool(os.environ.get(var))
    for var in OPTIONAL_ENV_VARS:
        results[var] = bool(os.environ.get(var))
    return results
