"""Round-robin over identical vLLM replicas (one per GPU) using two local HTTP stubs."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from clove.inference.backend import OpenAICompatBackend
from clove.inference.decode import decode
from clove.inference.prompt import build_prompt
from clove.memory.policy import Selection

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def stub(hits: list, tag: str):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj):
            b = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            self._send({"data": [{"id": MODEL}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            hits.append((tag, body))
            toks = ['{"', "answer", '":', ' "', "A", '"}']
            content = [{"token": t, "logprob": -0.01, "top_logprobs": [{"token": t, "logprob": -0.01}]} for t in toks]
            content[4]["top_logprobs"] = [{"token": "A", "logprob": -0.1}, {"token": "0", "logprob": -2.5}]
            self._send({"choices": [{"message": {"content": '{"answer": "A"}'}, "logprobs": {"content": content}}],
                        "usage": {"prompt_tokens": 1800, "completion_tokens": 6}})
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_round_robin_two_replicas_and_logprob_scores(tmp_path):
    from tests.conftest import make_image
    img = tmp_path / "q.jpg"
    make_image(img, 5)
    hits: list = []
    s1, s2 = stub(hits, "gpu0"), stub(hits, "gpu1")
    try:
        cfg = {"model_id": MODEL, "base_urls": [f"http://127.0.0.1:{s1.server_port}",
                                                f"http://127.0.0.1:{s2.server_port}"],
               "dtype": "float16", "max_pixels": 802816, "timeout_s": 10, "max_retries": 0}
        b = OpenAICompatBackend(cfg)
        assert b.health()[0]
        assert "base_url" not in b.describe() and b.describe()["dtype"] == "float16"
        sel = Selection(["Rust", "Brown_Spot"], {"Rust": [], "Brown_Spot": []}, 0, 0, "h")
        pr = build_prompt(sel)
        res = [b.infer(str(img), pr.system, pr.user_text, 96) for _ in range(4)]
        assert [t for t, _ in hits] == ["gpu0", "gpu1", "gpu0", "gpu1"]
        body = hits[0][1]
        imgs = [c for c in body["messages"][1]["content"] if c["type"] == "image_url"]
        assert len(imgs) == 1 and body["max_tokens"] == 96 and body["logprobs"] is True
        d = decode(res[0], pr)
        assert d.ok and d.label_id == "Rust" and d.score_source == "logprobs" and 0 < d.p_unknown < 0.2
        assert res[0].input_tokens == 1800
    finally:
        s1.shutdown(); s2.shutdown()
