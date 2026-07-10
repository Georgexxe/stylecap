"""Send one local frame to a Fireworks vision model as a cheap capability probe."""

import argparse
import base64
import mimetypes
import os
from pathlib import Path

from openai import OpenAI


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("frame", type=Path)
    args = parser.parse_args()

    mime = mimetypes.guess_type(args.frame.name)[0] or "image/jpeg"
    encoded = base64.b64encode(args.frame.read_bytes()).decode("ascii")
    client = OpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=os.environ["FIREWORKS_API_KEY"],
    )
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe only the main visible action in one sentence.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{encoded}"},
                    },
                ],
            }
        ],
        temperature=0,
        max_tokens=80,
        extra_body={"reasoning_effort": "none"},
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
