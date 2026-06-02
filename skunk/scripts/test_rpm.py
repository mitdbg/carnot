from skunk.retrieval.nodes.llm_wrapper import get_llm_wrapper


for model in [
    "google/gemini-3.1-pro-preview",
    "google/gemini-2.5-flash",
    "vertex_ai/gemini-3.1-flash",
    "vertex_ai/gemini-3.5-flash"]:
    prompts = [f'Tell me a joke based on the number {x}' for x in range(1000)]
    for x in range(1000):
        llm = get_llm_wrapper()

        try:
            response = llm.generate_google(
                prompt=prompts[x],
                model=model,
                system_prompt=''
            )
        except Exception as e:
            print("Model:", model)
            print(f"Error generating response for input {x}: {e}")
            break