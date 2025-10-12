import asyncio
import types
import json
import requests

from llm_adapters import OpenAIAdapter, LocalSubprocessAdapter, LocalStubAdapter


class DummyResponse:
    def __init__(self, text, json_obj=None):
        self._text = text
        self._json = json_obj

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=1024):
        # yield the text as one chunk
        yield self._text.encode('utf-8')

    def json(self):
        if self._json is not None:
            return self._json
        raise ValueError('no json')

    @property
    def text(self):
        return self._text


def monkeypatch_requests_post(monkey_text, json_obj=None):
    def _fake_post(*args, **kwargs):
        return DummyResponse(monkey_text, json_obj=json_obj)
    return _fake_post


def run_smoke():
    prompt = "Explain entropy in one sentence."

    # Test OpenAIAdapter with fake requests
    fake_json = {'choices': [{'text': 'Entropy measures disorder.'}]}
    requests.post = monkeypatch_requests_post('chunked-entropy', json_obj=fake_json)

    oa = OpenAIAdapter(api_key='fake')
    out = asyncio.run(oa.generate(prompt, max_tokens=50))
    print('OpenAIAdapter output:', out)

    # Test LocalSubprocessAdapter using `echo` (should return prompt)
    la = LocalSubprocessAdapter(cmd='echo')
    out2 = asyncio.run(la.generate(prompt))
    print('LocalSubprocessAdapter output (first 80 chars):', out2[:80])

    # Test LocalStubAdapter
    st = LocalStubAdapter('smoke-stub')
    out3 = asyncio.run(st.generate(prompt))
    print('LocalStubAdapter output:', out3[:80])


if __name__ == '__main__':
    run_smoke()
