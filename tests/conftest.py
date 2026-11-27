import logging

import pytest_asyncio

from agentic.common.logging import setup_logging
from agentic.devstack import Stack, make_settings

setup_logging("ERROR")
logging.getLogger("audit").setLevel(logging.ERROR)


@pytest_asyncio.fixture
async def stack():
    async with Stack() as s:
        yield s


@pytest_asyncio.fixture
async def bare_stack():
    """No knowledge seeding — faster, for gateway/trust tests."""
    async with Stack(seed_docs=False) as s:
        yield s


def stack_with(**over):
    """Helper for tests that need custom settings: `async with stack_with(rate_limit_per_minute=3) as s:`"""
    return Stack(make_settings(**over), seed_docs=False)
