# Creating Your Dataset

Learn how to create high-quality training datasets using nothing more than a simple text editor.
This practical guide walks through the entire process from conception to validation.

## Overview

Creating a good dataset is crucial for successful fine-tuning. You don't need complex tools.
A simple text editor and attention to detail are all you need to create professional-quality training data.

## Step-by-Step Process

### Step 1: Choose Your Text Editor

Any plain text editor will work:

=== "VS Code (Recommended)"
    ```bash
    # Install VS Code
    code data/my_custom_data.jsonl
    ```
    **Benefits**: JSON syntax highlighting, error detection, extensions

=== "Command Line Editors"
    ```bash
    # Nano (beginner-friendly)
    nano data/my_custom_data.jsonl

    # Vim (advanced)
    vim data/my_custom_data.jsonl

    # Emacs - Really? It's 2026 already.
    emacs data/my_custom_data.jsonl
    ```

=== "GUI Editors"
    - **Sublime**: Excellent JSON support
    - **Notepad++**: Windows-friendly with plugins

### Step 2: Set Up Your File Structure

Create the data directory and file:

    mkdir -p data
    touch data/my_custom_data.jsonl

### Step 3: Write Your First Example

Open your file and add the first training example:

    {"instruction": "What is hypertension?", "response": "Hypertension, commonly known as high blood pressure, is a medical condition where the blood pressure in the arteries is persistently elevated. It's defined as having a systolic pressure of 140 mmHg or higher, or a diastolic pressure of 90 mmHg or higher on multiple readings.", "source_urls": ["https://www.who.int/news-room/fact-sheets/detail/hypertension"]}

!!! important "One JSON Object Per Line"
    Each line must contain exactly one complete JSON object. No line breaks within the JSON object itself.

### Step 4: Add More Examples

Continue adding examples, one per line:

    {"instruction": "What is hypertension?", "response": "Hypertension, commonly known as high blood pressure, is a medical condition where the blood pressure in the arteries is persistently elevated...", "source_urls": ["https://www.who.int/news-room/fact-sheets/detail/hypertension"]}
    {"instruction": "How is diabetes diagnosed?", "response": "Diabetes is diagnosed using several blood tests. The most common methods include: fasting blood glucose test (≥126 mg/dL), oral glucose tolerance test (≥200 mg/dL after 2 hours), or HbA1c test (≥6.5%)...", "source_urls": ["https://www.diabetes.org/diabetes/a1c/diagnosis"]}
    {"instruction": "What are the symptoms of a heart attack?", "response": "Common heart attack symptoms include chest pain or discomfort, shortness of breath, pain or discomfort in the jaw, neck, back, arm, or shoulder, and feeling nauseous, light-headed, or unusually tired...", "source_urls": ["https://www.heart.org/en/health-topics/consumer-healthcare/what-is-cardiovascular-disease/warning-signs-of-heart-attack"]}

## Content Guidelines

### Medical AI Specific Considerations

For medical applications, ensure your responses:

1. **Include appropriate disclaimers**:

       "response": "Hypertension is... Always consult with a healthcare professional for medical advice."

2. **Avoid specific diagnoses**:

   ❌ "You have diabetes based on these symptoms"

   ✅ "These symptoms may indicate diabetes. Consult a healthcare provider for proper diagnosis."

3. **Focus on education**:

   ❌ "Take 10mg lisinopril daily"

   ✅ "ACE inhibitors like lisinopril are commonly prescribed for hypertension management"

4. **Express uncertainty when appropriate**:

       "response": "While chest pain can indicate heart issues, many conditions can cause chest pain.
       If experiencing chest pain, seek immediate medical attention."

### Response Quality Standards

#### Length and Detail

- **Concise but complete**: 2-4 sentences for simple questions
- **Detailed for complex topics**: Include context, mechanisms, considerations
- **Appropriate depth**: Match the complexity of the instruction

#### Tone and Style

```json
// Professional and helpful
{"instruction": "What is pneumonia?", "response": "Pneumonia is an infection that inflames the air sacs in one or both lungs, which may fill with fluid or pus. Common symptoms include cough with phlegm, fever, chills, and difficulty breathing. Treatment depends on the cause and severity, ranging from antibiotics for bacterial pneumonia to supportive care for viral cases."}

// Avoid overly casual tone
{"instruction": "What is pneumonia?", "response": "It's like when your lungs get really sick and infected and stuff..."}
```

### Source Documentation

Include credible sources when possible:

```json
{
  "instruction": "What are the latest hypertension guidelines?",
  "response": "The 2017 ACC/AHA guidelines define hypertension as blood pressure ≥130/80 mmHg, lowering the threshold from the previous 140/90 mmHg. This change emphasizes early intervention through lifestyle modifications and, when necessary, medication management.",
  "source_urls": [
    "https://www.ahajournals.org/doi/10.1161/HYP.0000000000000065",
    "https://www.acc.org/latest-in-cardiology/ten-points-to-remember/2017/11/09/11/41/2017-guideline-for-high-blood-pressure-in-adults"
  ]
}
```

## Validation and Testing

### JSON Validation

Use online validators or command-line tools:

```bash
# Using Python to validate JSON
python -c "
import json
with open('data/my_custom_data.jsonl', 'r') as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line.strip())
        except json.JSONDecodeError as e:
            print(f'Error on line {i}: {e}')
        else:
            print(f'Line {i}: Valid')
"
```

### Content Review Checklist

Before finalizing your dataset:

- [ ] All JSON objects are valid
- [ ] Each line contains one complete example
- [ ] Instructions are clear and specific
- [ ] Responses are accurate and helpful
- [ ] Medical disclaimers included where appropriate
- [ ] Sources cited for factual claims
- [ ] Consistent tone and style throughout
- [ ] Appropriate length and detail level

### Sample Size Guidelines

| Task Complexity | Minimum Examples | Recommended |
|-----------------|------------------|-------------|
| Simple Q&A | 20-50 | 100+ |
| Domain Adaptation | 50-100 | 200-500 |
| Style Transfer | 100-200 | 300-1000 |
| New Capabilities | 200-500 | 1000+ |

## Common Mistakes to Avoid

### Technical Issues

```json
// ❌ Line breaks in JSON
{
  "instruction": "What is diabetes?",
  "response": "Diabetes is a condition..."
}

// ✅ Single line JSON
{"instruction": "What is diabetes?", "response": "Diabetes is a condition..."}

// ❌ Missing commas or quotes
{"instruction": "What is diabetes?" "response": "Diabetes is a condition...}

// ✅ Proper JSON syntax
{"instruction": "What is diabetes?", "response": "Diabetes is a condition..."}
```

### Content Issues

```json
// ❌ Too brief
{"instruction": "Explain heart disease", "response": "It's bad for you."}

// ❌ Too prescriptive
{"instruction": "I have chest pain", "response": "You need to take aspirin and go to the ER immediately."}

// ✅ Educational and appropriate
{"instruction": "What should I know about chest pain?", "response": "Chest pain can have many causes, from minor muscle strain to serious heart conditions. If you experience chest pain, especially with symptoms like shortness of breath, nausea, or sweating, seek immediate medical attention."}
```

## Advanced Techniques

### Conversation Context

```json
{"instruction": "What is the normal blood pressure range?", "response": "Normal blood pressure is typically less than 120/80 mmHg. Elevated is 120-129 systolic with diastolic less than 80. Stage 1 hypertension is 130-139 systolic or 80-89 diastolic."}
{"instruction": "What if my blood pressure is 140/90?", "response": "A reading of 140/90 mmHg indicates Stage 2 hypertension. This requires medical evaluation and likely treatment. Multiple readings should be taken over time to confirm the diagnosis, as blood pressure can vary throughout the day."}
```

### Difficulty Progression

Start with simple examples and gradually increase complexity:

1. **Basic facts**: "What is diabetes?"
2. **Explanations**: "How does insulin work?"
3. **Applications**: "How do you manage diabetes with diet?"
4. **Complex scenarios**: "What are the long-term complications of uncontrolled diabetes?"

---

You now have a complete understanding of dataset creation for LoRA fine-tuning!
