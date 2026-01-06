# Understanding JSONL Format

JSONL (JSON Lines) is a text format where each line is a separate, valid JSON object. It's perfect for machine learning datasets because it's easy to read, write, and process line-by-line.

## What is JSONL?

**JSONL** stands for **JSON Lines**. Unlike regular JSON files that contain a single JSON object or array, JSONL files contain multiple JSON objects, each on its own line.

### Regular JSON vs JSONL

=== "Regular JSON"
    ```json
    {
      "data": [
        {
          "instruction": "What is diabetes?",
          "response": "Diabetes is a metabolic disorder..."
        },
        {
          "instruction": "How is blood pressure measured?",
          "response": "Blood pressure is measured using..."
        }
      ]
    }
    ```

=== "JSONL Format"
    ```jsonl
    {"instruction": "What is diabetes?", "response": "Diabetes is a metabolic disorder..."}
    {"instruction": "How is blood pressure measured?", "response": "Blood pressure is measured using..."}
    ```

## Why Use JSONL for ML?

### Memory Efficiency

- **Streaming**: Process one example at a time without loading entire file
- **Large datasets**: Handle files that don't fit in memory
- **Parallel processing**: Easy to split across workers

### Simplicity

- **Append-only**: Add new examples by appending lines
- **Error isolation**: One corrupt line doesn't break the entire file
- **Human-readable**: Easy to inspect and edit

### Tool Support

- **Pandas**: `pd.read_json(lines=True)`
- **Hugging Face**: `load_dataset("json", data_files="file.jsonl")`
- **Command line**: `cat`, `head`, `tail`, `wc -l` work naturally

## JSONL Structure for Training

### Required Fields

Each line must be a JSON object with these fields:

```jsonl
{"instruction": "Your question or task", "response": "Expected model response"}
```

### Optional Fields

Enhance your dataset with additional metadata:

```jsonl
{
  "instruction": "What are the symptoms of hypertension?",
  "response": "High blood pressure often has no symptoms, earning it the nickname 'silent killer'. Some people may experience headaches, shortness of breath, or nosebleeds, but these signs usually don't occur until blood pressure reaches dangerously high levels.",
  "source_urls": ["https://www.mayoclinic.org/diseases-conditions/high-blood-pressure/symptoms-causes/syc-20373410"],
  "difficulty": "intermediate",
  "category": "cardiovascular",
  "medical_specialty": "cardiology",
  "confidence": 0.95
}
```

Common optional fields:

| Field | Purpose | Example |
|-------|---------|---------|
| `source_urls` | Reference materials | `["https://example.com"]` |
| `category` | Topic classification | `"cardiovascular"` |
| `difficulty` | Complexity level | `"beginner"`, `"advanced"` |
| `medical_specialty` | Domain area | `"cardiology"`, `"neurology"` |
| `confidence` | Response quality score | `0.8` (0-1 scale) |
| `tags` | Keywords | `["diabetes", "diagnosis"]` |

## Creating Valid JSONL

### Basic Rules

1. **One JSON object per line**
2. **No line breaks within JSON objects**
3. **Valid JSON syntax on each line**
4. **UTF-8 encoding recommended**

### Example Creation Process

#### Step 1: Start Simple

```jsonl
{"instruction": "What is a heart attack?", "response": "A heart attack occurs when blood flow to part of the heart muscle is blocked, usually by a blood clot in a coronary artery."}
```

#### Step 2: Add More Detail

```jsonl
{"instruction": "What is a heart attack?", "response": "A heart attack (myocardial infarction) occurs when blood flow to part of the heart muscle is blocked, usually by a blood clot in a coronary artery. This lack of blood flow damages or destroys part of the heart muscle. Common symptoms include chest pain, shortness of breath, nausea, and sweating. Immediate medical attention is critical for the best outcomes."}
```

#### Step 3: Include Sources

```jsonl
{"instruction": "What is a heart attack?", "response": "A heart attack (myocardial infarction) occurs when blood flow to part of the heart muscle is blocked, usually by a blood clot in a coronary artery. This lack of blood flow damages or destroys part of the heart muscle. Common symptoms include chest pain, shortness of breath, nausea, and sweating. Immediate medical attention is critical for the best outcomes.", "source_urls": ["https://www.heart.org/en/health-topics/heart-attack"]}
```

## Data Processing Pipeline

### Loading JSONL in Code

=== "Python (Hugging Face)"
    ```python
    from datasets import load_dataset

    # Load JSONL file
    dataset = load_dataset("json", data_files="data/my_custom_data.jsonl")
    print(dataset["train"][0])  # First example
    ```

=== "Python (Manual)"
    ```python
    import json

    examples = []
    with open("data/my_custom_data.jsonl", "r") as f:
        for line in f:
            examples.append(json.loads(line.strip()))

    print(f"Loaded {len(examples)} examples")
    ```

=== "Pandas"
    ```python
    import pandas as pd

    df = pd.read_json("data/my_custom_data.jsonl", lines=True)
    print(df.head())
    ```

### Command Line Tools

```bash
# Count examples
wc -l data/my_custom_data.jsonl

# View first 5 examples
head -5 data/my_custom_data.jsonl

# View last 5 examples
tail -5 data/my_custom_data.jsonl

# Search for specific content
grep -i "diabetes" data/my_custom_data.jsonl

# Validate JSON syntax
cat data/my_custom_data.jsonl | python -m json.tool > /dev/null && echo "Valid JSON"
```

## Quality Validation

### Automated Checks

Create a validation script:

```python
import json
import sys

def validate_jsonl(file_path):
    errors = []
    required_fields = ["instruction", "response"]

    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON - {e}")
                continue

            # Check required fields
            for field in required_fields:
                if field not in obj:
                    errors.append(f"Line {line_num}: Missing required field '{field}'")

            # Check field types
            if "instruction" in obj and not isinstance(obj["instruction"], str):
                errors.append(f"Line {line_num}: 'instruction' must be a string")

            if "response" in obj and not isinstance(obj["response"], str):
                errors.append(f"Line {line_num}: 'response' must be a string")

    if errors:
        print(f"Found {len(errors)} errors:")
        for error in errors:
            print(f"  {error}")
        return False
    else:
        print("✅ All validation checks passed!")
        return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate.py <jsonl_file>")
        sys.exit(1)

    validate_jsonl(sys.argv[1])
```

### Manual Review Checklist

- [ ] Each line contains valid JSON
- [ ] Required fields present in all examples
- [ ] No line breaks within JSON objects
- [ ] Consistent field naming and types
- [ ] Appropriate content length
- [ ] No duplicate examples
- [ ] Balanced representation of topics

## Advanced JSONL Techniques

### Streaming Processing

For large datasets, process line-by-line:

```python
def process_large_jsonl(file_path):
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f):
            if line_num % 10000 == 0:
                print(f"Processed {line_num} examples...")

            example = json.loads(line.strip())
            # Process example here
            yield example
```

### Parallel Processing

Split processing across CPU cores:

```python
from multiprocessing import Pool
import json

def process_batch(lines):
    results = []
    for line in lines:
        example = json.loads(line.strip())
        # Process example
        results.append(example)
    return results

def parallel_process_jsonl(file_path, num_workers=4):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Split into batches
    batch_size = len(lines) // num_workers
    batches = [lines[i:i+batch_size] for i in range(0, len(lines), batch_size)]

    with Pool(num_workers) as pool:
        results = pool.map(process_batch, batches)

    # Flatten results
    return [item for batch in results for item in batch]
```

## Common Mistakes

### JSON Syntax Errors

```jsonl
// ❌ Line breaks in JSON object
{
  "instruction": "What is diabetes?",
  "response": "A condition..."
}

// ❌ Missing quotes
{"instruction": What is diabetes?, "response": "A condition..."}

// ❌ Trailing comma
{"instruction": "What is diabetes?", "response": "A condition...",}

// ✅ Correct format
{"instruction": "What is diabetes?", "response": "A condition..."}
```

### Field Issues

```jsonl
// ❌ Inconsistent field names
{"question": "What is diabetes?", "answer": "A condition..."}
{"instruction": "What is hypertension?", "response": "High blood pressure..."}

// ✅ Consistent field names
{"instruction": "What is diabetes?", "response": "A condition..."}
{"instruction": "What is hypertension?", "response": "High blood pressure..."}
```

---

Next: Learn the complete process of [creating your dataset](creating-dataset.md) using simple text editors.
