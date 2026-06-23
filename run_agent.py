from agent import Agent
import argparse

def main():
     parser = argparse.ArgumentParser(
         description="Evaluate scientific claims using LLM models",
         formatter_class=argparse.RawDescriptionHelpFormatter,
     )

     parser.add_argument(
         '-i', '--input-file',
         type=str,
         required=True,
         help='Path to the input file (JSON format)'
     )

     parser.add_argument(
         '-m', '--model',
         type=str,
         required=True,
         help='Path to the model directory'
     )

     parser.add_argument(
         '-o', '--output-file',
         type=str,
         required=True,
         help='Path to the output file where results will be saved'
     )

     parser.add_argument(
         '-s', '--sentiment-model',
         type=str,
         default=None,
         help='Path to the (optional) model for sentiment analysis'
     )

     args = parser.parse_args()

     agent = Agent(
         model_path=args.model,
         sentiment_model_path=args.sentiment_model
     )

     agent.evaluate_claims(
         input_file=args.input_file,
         output_file=args.output_file
     )

if __name__ == "__main__":
    main()