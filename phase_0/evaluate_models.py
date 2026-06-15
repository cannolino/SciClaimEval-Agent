from openai import OpenAI
from pathlib import Path
import argparse, base64, json

def main():
    api_key = '<api-key>'
    base_url = "https://chat-ai.academiccloud.de/v1"
    # with open("phase_0/models_list.txt", "r") as f:
    #     models = f.read()
    #     print(models.splitlines())
    model = ["internvl3.5-30b-a3b"]
    client = OpenAI(
        api_key = api_key,
        base_url = base_url
    )
    parser = argparse.ArgumentParser(
        description="Evaluate scientific claims using LLM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '-i', '--input-file',
        type=str,
        required=True,
        help='Path to the input file (JSON format) containing the dataset'
    )
    parser.add_argument(
        '-o', '--output-file',
        type=str,
        required=True,
        help='Path to the output file where evaluation results will be saved'
    )
    args = parser.parse_args()
    input_file = Path(args.input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_responses(input_file, model, client, output_file)

def encode_image(image_path):
    print(image_path)
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def write_responses(input_file, models, client:OpenAI, output_file):
    dic = dict()
    with open(input_file, "r") as f:
        dataset = json.load(f)
    for model in models:
        dic[model] = list()
        for data in dataset[:10]:
            prompt = 'Determine whether the claim: "' + data["claim"] + '" is supported or refuted, given evidence depicted in the image.'
            # TODO: adjust prompt given context information accordingly
            # if data["use_context"] == "no":
            #     prompt =
            # if data["use_context"] == "yes":
            #     prompt =
            # if data["use_context"] == "other sources":
            #     prompt =
            base64_image = encode_image(input_file.parent / data["evi_path"])
            response = client.chat.completions.create(
                    max_completion_tokens=1,
                    model=model,
                    messages=[
                        {"role":"system","content":"You are a scientific reviewer."},
                        {
                            "role":"user",
                            "content": [
                                { "type": "text", "text": prompt },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                )
            dic[model].append({"claim_id": data["claim_id"], "pred_label": response.choices[0].message.content})
    with open(output_file, "w") as f:
        json.dump(dic, f, indent=2)

if __name__ == "__main__":
    main()