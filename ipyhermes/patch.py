from fastcore.all import *
from run_agent import AIAgent
import asyncio

@patch
async def astream(self:AIAgent, prompt:str, sp:str=None, hist:list=None):
    "Async generator: yields text deltas. After exhaustion, access full text via astream.last_text and run_conversation result via astream.last_result."
    loop = asyncio.get_running_loop()
    q, sentinel = asyncio.Queue(), object()
    _orig = self.stream_delta_callback
    self.stream_delta_callback = lambda chunk: loop.call_soon_threadsafe(q.put_nowait, chunk)
    fut = loop.run_in_executor(None,
        lambda: self.run_conversation(user_message=prompt, system_message=sp,
                                      conversation_history=hist or None))
    fut.add_done_callback(lambda _: loop.call_soon_threadsafe(q.put_nowait, sentinel))
    full = []
    try:
        while True:
            item = await q.get()
            if item is sentinel: break
            full.append(item); yield item
    finally:
        self.stream_delta_callback = _orig
    astream.last_result = await fut
    astream.last_text = ''.join(full)
