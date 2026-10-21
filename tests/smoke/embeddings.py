"""Opt-in multimodal embedding checks against an already running isolated server.

uv run python tests/smoke/embeddings.py --server http://127.0.0.1:8991 --model ID
Requires ffmpeg and Pillow (already in the runtime); uses temporary synthetic
media. Checks transport/encoder sensitivity, not retrieval quality certification.
"""
import argparse
import base64
import io
import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile
import urllib.request
import wave

from PIL import Image


def fixtures(folder):
    def block(kind, mime, data):
        return {"type": kind, "source": {"type": "base64", "media_type": mime,
                                         "data": base64.b64encode(data).decode()}}
    images, audios, videos = [], [], []
    for color in ('red', 'blue'):
        out = io.BytesIO()
        Image.new('RGB', (128, 128), color).save(out, format='PNG')
        images.append(block('image', 'image/png', out.getvalue()))
        path = Path(folder) / f'{color}.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                        f'color=c={color}:s=128x128:r=2:d=2', '-pix_fmt', 'yuv420p', str(path)], check=True)
        videos.append(block('video', 'video/mp4', path.read_bytes()))
    for hz in (440, 880):
        out = io.BytesIO()
        with wave.open(out, 'wb') as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
            wav.writeframes(b''.join(struct.pack('<h', int(10000 * math.sin(2*math.pi*hz*i/16000))) for i in range(16000)))
        audios.append(block('audio', 'audio/wav', out.getvalue()))
    text = lambda value: {"type": "text", "text": value}
    return [
        {"content": [text(t)]} for t in ('A dog playing on a beach.', 'A puppy running beside the ocean.', 'A database query joining two tables.')
    ] + [{"content": [b]} for b in images + audios + videos] + [
        {"content": [text('A scene with sound.'), images[i], audios[0], videos[0]]} for i in range(2)
    ]


def post(server, model, items, dimensions=None):
    body = {"model": model, "input": items}
    if dimensions: body['dimensions'] = dimensions
    request = urllib.request.Request(server.rstrip('/')+'/v1/embeddings', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'{exc.code}: {exc.read().decode()}') from exc


def cosine(a, b):
    return math.fsum(x*y for x,y in zip(a,b))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='embedding-smoke-') as folder:
        items = fixtures(folder)
        result = post(args.server, args.model, items)
        vectors = [row['embedding'] for row in result['data']]
        assert len(vectors) == len(items)
        assert [row['index'] for row in result['data']] == list(range(len(items)))
        for vector in vectors:
            assert len(vector) == 768 and all(map(math.isfinite, vector))
            assert abs(cosine(vector, vector)-1) < 1e-6
        assert cosine(vectors[0], vectors[1]) > cosine(vectors[0], vectors[2])
        for left,right in ((3,4),(5,6),(7,8),(9,10)):
            assert cosine(vectors[left],vectors[right]) < .99999, f'media ignored: {left}/{right}'
        single = post(args.server,args.model,[items[3]])['data'][0]['embedding']
        assert cosine(single,vectors[3]) > .9999, 'batch/singleton mismatch'
        for dimension in (128,256,512):
            short = post(args.server,args.model,[items[0]],dimension)['data'][0]['embedding']
            expected = vectors[0][:dimension]
            norm = math.sqrt(cosine(expected,expected))
            assert cosine(short,[x/norm for x in expected]) > .9999
        with urllib.request.urlopen(args.server.rstrip('/')+'/v1/models') as response:
            row = next(m for m in json.load(response)['data'] if m['id'] == args.model)
        for name, prefix in row['engine']['embedding']['prompts'].items():
            text = 'A dog playing on a beach.'
            prompted = post(args.server, args.model, [
                {'prompt_name': name, 'content': [{'type': 'text', 'text': text}]}, prefix + text,
            ])['data']
            assert cosine(prompted[0]['embedding'], prompted[1]['embedding']) > .9999, f'prefix mismatch: {name}'
        # Refusal must happen before a forward silently truncates the item.
        try:
            post(args.server, args.model, ['hello ' * 10000])
        except RuntimeError as exc:
            assert str(exc).startswith('400:'), str(exc)
        else:
            raise AssertionError('over-context input was accepted')
        if args.output: Path(args.output).write_text(json.dumps(result))
        print(f'PASS {args.model}: text ranking, image/audio/video/joint sensitivity, singleton agreement, dimensions, finite unit vectors, all named prefixes, context refusal', flush=True)


if __name__ == '__main__':
    main()
