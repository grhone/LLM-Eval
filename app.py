import os
import time
import pandas as pd
import yaml
from langchain_community.chat_models import ChatOllama, ChatGooglePalm, ChatOpenAI, ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from tqdm import tqdm

from utils.prompts import get_judge_prompt, get_zeroshot_qa_prompt

def load_config(config_path="config.yaml"):
    """Load and parse YAML configuration file.

    Args:
        config_path (str, optional): Path to YAML config file. Defaults to "config.yaml".

    Returns:
        dict: Parsed configuration as dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
        yaml.YAMLError: If config file contains invalid YAML.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def create_model(model_config):
    """Initialize an LLM model instance from configuration.

    Args:
        model_config (dict): Configuration dictionary containing:
            - provider (str): Model provider (openai, anthropic, ollama, gemini, openrouter)
            - model_name (str): Name of model to use
            - api_key (str): API key for provider (if required)
            - base_url (str): Base URL for self-hosted models (if required)

    Returns:
        BaseChatModel: Initialized LangChain chat model instance.

    Raises:
        ValueError: If provider is not supported.
    """
    provider = model_config["provider"]
    model_name = model_config["model_name"]

    if provider == "openai":
        return ChatOpenAI(model_name=model_name, openai_api_key=model_config["api_key"])
    elif provider == "anthropic":
        return ChatAnthropic(model_name=model_name, anthropic_api_key=model_config["api_key"])
    elif provider == "ollama":
        return ChatOllama(model=model_name, base_url=model_config["base_url"], num_ctx=5000, num_predict=1000)
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=model_config["api_key"], max_tokens=2000)
    elif provider == "openrouter":
        return ChatOpenAI(
            model_name=model_name,
            openai_api_key=model_config["api_key"],
            openai_api_base="https://openrouter.ai/api/v1",
        )
    else:
        raise ValueError(f"Unsupported model provider: {provider}")

def main():
    """Run benchmark evaluation pipeline for LLM models.

    Workflow:
    1. Load configuration from config.yaml
    2. Initialize judge and benchmark models
    3. Load evaluation dataset from parquet file
    4. For each benchmark model:
        a. Generate answers for all questions
        b. Evaluate answers using judge model
    5. Save results to CSV and print summary statistics

    Outputs:
        Creates benchmark_results.csv with columns:
        - model_provider: Provider name
        - model_name: Model name
        - question: Evaluation question
        - gold_answer: Ground truth answer
        - model_answer: Generated answer
        - judge_score: Evaluation score (0 or 1)

    Side Effects:
        - Prints progress updates during evaluation
        - Prints final accuracy statistics per model
    """
    # Load configuration
    config = load_config()
    judge_model_config = config["judge_model"]
    benchmark_models_config = config["benchmark_models"]

    # Create the judge model
    judge_model = create_model(judge_model_config)

    # Load the data
    #TODO: Make this dynamic
    df = pd.read_parquet("data/train-00000-of-00001.parquet")

    results = []

    # Loop through each model to be benchmarked
    for model_config in benchmark_models_config:
        print(f"Benchmarking model: {model_config['provider']}/{model_config['model_name']}")
        model_to_benchmark = create_model(model_config)

        # Loop through each row in the dataset
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            question = row["question"]
            gold_answer = row["ground_truth_answer"]
            document_summary = row.get("document_summary", "")  # Optional field
            chunks = row.get("chunks", [])  # Optional field
            combined_chunks = "\n".join(chunks) if isinstance(chunks, list) else ""

            # Generate an answer using the model to be benchmarked
            try:
                qa_prompt = get_zeroshot_qa_prompt(question)
                raw_answer = model_to_benchmark.predict(qa_prompt)
                # Extract the answer from the XML tags
                model_answer = raw_answer.split("<answer>")[1].split("</answer>")[0].strip()
            except Exception as e:
                model_answer = f"Error generating answer: {e}"

            # Use the judge model to evaluate the answer
            judge_prompt = get_judge_prompt(
                question=question,
                answer=model_answer,
                gold=gold_answer,
                chunks=combined_chunks,
                documents=document_summary,
            )
            
            final_answer = "Error: Max retries reached"
            for attempt in range(3):  # Retry up to 3 times
                try:
                    judge_response = judge_model.invoke(judge_prompt)
                    # Extract the final answer from the XML tags
                    final_answer = judge_response.content.split("<final_answer>")[1].split("</final_answer>")[0].strip()
                    break  # Success, exit loop
                except IndexError:
                    print(f"Attempt {attempt + 1} failed: list index out of range. Full response: {judge_response.content}")
                    time.sleep(2)  # Wait for 2 seconds before retrying
                except Exception as e:
                    final_answer = f"Error judging answer: {e}"
                    break


            results.append(
                {
                    "model_provider": model_config["provider"],
                    "model_name": model_config["model_name"],
                    "question": question,
                    "gold_answer": gold_answer,
                    "model_answer": model_answer,
                    "judge_score": final_answer,
                }
            )

    # Save the results to a CSV file
    results_df = pd.DataFrame(results)
    #TODO: Make this dynamic
    results_df.to_csv("benchmark_results.csv", index=False)
    print("Benchmarking complete. Results saved to benchmark_results.csv")

    # Calculate and display the results
    print("\n--- Benchmark Results ---")
    for model_config in benchmark_models_config:
        model_name = model_config["model_name"]
        model_results = results_df[results_df["model_name"] == model_name]
        
        if not model_results.empty:
            # Convert 'judge_score' to numeric, coercing errors to NaN
            model_results["judge_score"] = pd.to_numeric(model_results["judge_score"], errors='coerce')
            
            # Drop rows where 'judge_score' is NaN
            model_results.dropna(subset=['judge_score'], inplace=True)
            
            if not model_results.empty:
                total_questions = len(model_results)
                correct_answers = model_results["judge_score"].sum()
                accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0

                print(f"\nModel: {model_name}")
                print(f"  - Total Questions: {total_questions}")
                print(f"  - Correct Answers: {int(correct_answers)}")
                print(f"  - Accuracy: {accuracy:.2f}%")
            else:
                print(f"\nModel: {model_name}")
                print("  - No valid judge scores found.")
        else:
            print(f"\nModel: {model_name}")
            print("  - No results found for this model.")

if __name__ == "__main__":
    main()
