from llm_wrapper import parse_json_response

x = """
```thought
```json
{
  "text_blocks": [
    {
      "block_idx": 0,
      "description": "Reporting requirements for securities brokers and dealers regarding their own and custody liabilities to foreigners."
    },
    {
      "block_idx": 1,
      "description":"Revisions and restructuring of Table CM"}]}
```
"""

print(parse_json_response(x))