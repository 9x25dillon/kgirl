import asyncio
import os

os.environ.setdefault("JULIA_SERVER_URL", "http://localhost:8088")
os.environ.setdefault("JULIA_WS_URL", "ws://localhost:8089")

cursor/bc-f408c7bd-bc2a-48a4-bc8d-0989f628ad52-ef2e
from chaos_llm.services.al_uls_client import al_uls_client
from chaos_llm.services.al_uls_ws_client import al_uls_ws_client
from src.chaos_llm.services.al_uls_client import al_uls_client
from src.chaos_llm.services.al_uls_ws_client import al_uls_ws_client
main

async def main():
    print("HTTP health:", await al_uls_client.health())
    res1 = await al_uls_client.eval("SUM", ["1","2","3"]) ; print("HTTP SUM:", res1)
    res2 = await al_uls_ws_client.eval("MEAN", ["4","5","6"]) ; print("WS MEAN:", res2)
    batch = await al_uls_ws_client.batch_eval([
        {"name":"SUM","args":["1","2","3"]},
        {"name":"VAR","args":["10","20","30"]}
    ])
    print("WS batch:", batch)

if __name__ == "__main__":
    asyncio.run(main())
