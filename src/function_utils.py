"""Function calling and tool execution utilities."""

import re
import json
from typing import List, Dict, Any, Tuple


def extract_function_calls(response_text: str) -> List[Dict[str, Any]]:
    """Extract function calls from model response text."""
    all_calls = []
    
    # Try JSON format first
    json_pattern = r'\[{.*?}\]'
    json_matches = re.findall(json_pattern, response_text, re.DOTALL)
    
    for match in json_matches:
        try:
            function_calls = json.loads(match)
            all_calls.extend(function_calls)
        except json.JSONDecodeError:
            continue
    
    # Also try function call format
    func_pattern = r'(\w+)\((.*?)\)'
    func_matches = re.findall(func_pattern, response_text)
    
    for func_name, args_str in func_matches:
        if func_name in ["search_internet", "retrieve_clinical_guidelines"]:
            call = {"name": func_name, "arguments": {}}
            
            # Extract query parameter
            if "query=" in args_str:
                query_match = re.search(r'query="([^"]*)"', args_str)
                if query_match:
                    call["arguments"]["query"] = query_match.group(1)
            
            # Extract other parameters
            if "num_results=" in args_str:
                num_match = re.search(r'num_results=(\d+)', args_str)
                if num_match:
                    call["arguments"]["num_results"] = int(num_match.group(1))
            
            if "condition=" in args_str:
                condition_match = re.search(r'condition="([^"]*)"', args_str)
                if condition_match:
                    call["arguments"]["condition"] = condition_match.group(1)
            
            if "guideline_source=" in args_str:
                source_match = re.search(r'guideline_source="([^"]*)"', args_str)
                if source_match:
                    call["arguments"]["guideline_source"] = source_match.group(1)
            
            all_calls.append(call)
    
    return all_calls


def execute_function_call(function_name: str, arguments: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Execute a function call and return result with source URLs."""
    source_urls = []
    
    if function_name == "search_internet":
        query = arguments.get("query", "")
        result = f"""Recent search results for '{query}':
1. American Heart Association 2023 Guidelines: New BP targets <130/80 for most adults
2. NICE Guidelines (Updated 2022): ACE inhibitors or ARBs as first-line therapy
3. ESC/ESH 2023 Consensus: Emphasis on lifestyle modifications before medication
4. Recent meta-analysis shows combination therapy reduces cardiovascular events by 25%
5. 2023 studies highlight importance of home BP monitoring for treatment decisions"""
        
        source_urls.extend([
            "https://www.ahajournals.org/doi/10.1161/HYP.0000000000000217",
            "https://www.nice.org.uk/guidance/ng136",
            "https://academic.oup.com/eurheartj/article/44/31/2971/7243215"
        ])
        
    elif function_name == "retrieve_clinical_guidelines":
        condition = arguments.get("condition", "")
        source = arguments.get("guideline_source", "general")
        
        result = f"""Clinical Guidelines for {condition} from {source}:

NICE Guidelines for Hypertension (2023 Update):
• Blood pressure targets: <140/90 mmHg for most patients, <130/80 for high CVD risk
• First-line treatment: ACE inhibitor or ARB for patients <55 years or diabetes
• For patients ≥55 years or African/Caribbean: Calcium channel blocker first-line
• Step 2: Add calcium channel blocker or thiazide-like diuretic
• Step 3: Triple therapy with ACE inhibitor/ARB + CCB + diuretic
• Step 4: Add spironolactone or higher dose diuretic
• Lifestyle: <6g salt/day, BMI 20-25, regular exercise, limit alcohol
• Annual review with BP monitoring and cardiovascular risk assessment"""
        
        if source.upper() == "NICE":
            url = "https://www.nice.org.uk/guidance"
            source_urls.append(url)
        elif source.upper() == "AHA":
            url = "https://www.ahajournals.org/guidelines"
            source_urls.append(url)
        else:
            url = "https://www.who.int/publications/guidelines"
            source_urls.append(url)
    
    else:
        result = f"Function {function_name} not implemented in this demo"
    
    return result, source_urls


def request_permission(function_name: str, arguments: Dict[str, Any]) -> List[str]:
    """Request user permission to access tools and return source URLs."""
    source_urls = []
    
    if function_name == "search_internet":
        query = arguments.get("query", "")
        print(f"🔍 I need to search the internet for: '{query}'")
        print("   This will access medical research databases and clinical resources.")
        source_urls.extend([
            "https://www.ahajournals.org/doi/10.1161/HYP.0000000000000217",
            "https://www.nice.org.uk/guidance/ng136",
            "https://academic.oup.com/eurheartj/article/44/31/2971/7243215"
        ])
        
    elif function_name == "retrieve_clinical_guidelines":
        condition = arguments.get("condition", "")
        source = arguments.get("guideline_source", "general")
        
        if source.upper() == "NICE":
            url = "https://www.nice.org.uk/guidance"
            print(f"🏥 I need to check the website: {url}")
            print(f"   To retrieve {source} clinical guidelines for {condition}")
            source_urls.append(url)
        elif source.upper() == "AHA":
            url = "https://www.ahajournals.org/guidelines"
            print(f"🏥 I need to check the website: {url}")
            print(f"   To retrieve {source} clinical guidelines for {condition}")
            source_urls.append(url)
        else:
            url = "https://www.who.int/publications/guidelines"
            print(f"🏥 I need to check the website: {url}")
            print(f"   To retrieve clinical guidelines for {condition}")
            source_urls.append(url)
    
    print("   Proceeding with search...\n")
    return source_urls


def has_function_calls(response_text: str) -> bool:
    """Check if response contains function calls."""
    return (("[{" in response_text and '"name"' in response_text) or 
            ("search_internet(" in response_text or "retrieve_clinical_guidelines(" in response_text))